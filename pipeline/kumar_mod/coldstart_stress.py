"""Stress test the 'conditioning on public rules is inert' null for the cold-start encoder, to settle the
skeptic's attack: maybe the bilinear was just too weak/regularized to use the rules. Battery:

  1. RANK sweep      : rank in {1,4,8,16,32,64} (does more capacity ever let rules help?)
  2. REG sweep       : weight_decay in {1e-4,1e-3,1e-2} (is the null robust to the penalty, not one bad value?)
  3. TEXT variants   : community vector from name / description / rules / rules+description
  4. WRONG-rules x K  : 20 random wrong-community assignments (mean+/-CI of right-minus-wrong, not one draw)
  5. POSITIVE CONTROL: condition on a LABEL-derived prototype z = class 1 (removed) minus class 0 (kept) over the
                       community's train rows, i.e. mean(yy==1) - mean(yy==0). If the
                       SAME architecture USES this (>> global), the architecture works -> a rules-null is genuine.

All vs the no-conditioning global head, leave-one-community-out, on the balanced test fold (same protocol/metrics
as coldstart_conditional_encoder). CPU-only, parallel (<=75% cores). Out: results/kumar_mod/coldstart_stress.json
  smoke: python -m pipeline.kumar_mod.coldstart_stress --smoke
  run:   python -m pipeline.kumar_mod.coldstart_stress --jobs 16
"""
from __future__ import annotations
import os
import argparse, json, os, sys
from pathlib import Path
# Pin BLAS to a single thread per process so it doesn't fight the joblib process pool for cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K

BAL = ROOT / "results" / "kumar_mod" / "balanced"
ECACHE = BAL / "_fairness_e5_cache.npy"
TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
OUT = ROOT / "results" / "kumar_mod" / "coldstart_stress.json"
JTMP = ROOT / ".joblib_tmp"
ENC_ID, ENC_PREFIX = "intfloat/e5-large-v2", "query: "
# DZ: community-vector dim after PCA. SEED: fixed for every fit (torch + numpy) so the whole battery is
# reproducible. KWRONG: number of random wrong-community assignments per fold (mean+/-CI, not one draw).
DZ, SEED, EPOCHS, KWRONG = 32, 11, 100, 20
# Capacity (rank) and regularisation (weight_decay) grids: the skeptic's question is whether ANY cell ever
# lets the rules beat the no-conditioning global head. If none does, the null survives the whole grid.
RANKS, WDS = [1, 4, 8, 16, 32, 64], [1e-4, 1e-3, 1e-2]
TEXT_KINDS = ["name", "description", "rules", "rules_desc"]
_XMM = None


def _Xmm():
    # Memory-map the e5 embedding cache so each worker process shares the file rather than copying it.
    global _XMM
    if _XMM is None:
        _XMM = np.load(ECACHE, mmap_mode="r")
    return _XMM


def _fit_lr(X, y):
    # The global head: a plain LR on pooled cross-community data, no conditioning. Guard against a
    # single-class fold (roc_auc is undefined there).
    return LogisticRegression(C=1.0, max_iter=300).fit(X, y) if len(np.unique(y)) == 2 else None


def _bilinear(Xtr, ytr, ztr, rank, wd, epochs=EPOCHS, bs=4096, lr=1e-3):
    # Low-rank bilinear head: the per-community weight is w0 + B(Qz), so the community vector z modulates
    # the linear predictor through a rank-`rank` bottleneck. This is the architecture under attack -- if it
    # can't use z, we need to know whether that's the architecture's fault or because z carries no signal.
    import torch
    torch.manual_seed(SEED); torch.set_num_threads(1)
    d, dz = Xtr.shape[1], ztr.shape[1]
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Zt = torch.tensor(ztr, dtype=torch.float32)
    w0 = torch.zeros(d, requires_grad=True)
    # B, Q are the low-rank factors of the conditioning interaction; small-std init keeps the model near the
    # global head at start so conditioning only helps if the data pushes it to.
    B = torch.empty(d, rank); torch.nn.init.normal_(B, std=0.02); B.requires_grad_(True)
    Q = torch.empty(rank, dz); torch.nn.init.normal_(Q, std=0.02); Q.requires_grad_(True)
    v = torch.zeros(dz, requires_grad=True); b0 = torch.zeros(1, requires_grad=True)
    # weight_decay is the reg knob swept by WDS; it penalises B/Q too, so the sweep tests whether the null
    # is an artifact of one penalty value rather than a real absence of usable signal.
    opt = torch.optim.Adam([w0, B, Q, v, b0], lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss(); n = Xt.shape[0]; g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            ix = perm[i:i + bs]; xb, yb, zb = Xt[ix], yt[ix], Zt[ix]
            # logit = global term + bilinear interaction (x.B times z.Q, summed over rank) + z bias + intercept.
            logit = xb @ w0 + (xb @ B * (zb @ Q.T)).sum(1) + zb @ v + b0
            opt.zero_grad(); lossf(logit, yb).backward(); opt.step()
    with torch.no_grad():
        return (w0.numpy(), B.numpy(), Q.numpy(), v.numpy(), float(b0.item()))


def _wb(p, z):
    # Collapse the trained bilinear into an effective linear (weight, bias) for a held-out community's z.
    # This is what lets a community never seen at fit time get its own head purely from its text vector.
    w0, B, Q, v, b0 = p
    return w0 + B @ (Q @ z), float(v @ z + b0)


def _auc(sc, y):
    return float(roc_auc_score(y, sc)) if len(np.unique(y)) == 2 else None


def _comm_text(comms, kind):
    # Build the text that gets encoded into each community vector, one of four variants. The TEXT sweep asks
    # whether the rules-null is specific to rules or holds for name/description too -- and whether adding
    # rules to the description ever helps over the description alone.
    desc, rules = K.load_rules()
    out = []
    for s in comms:
        rs = rules.get(s, "")
        if isinstance(rs, (list, tuple)):
            rs = " ".join(str(x) for x in rs)
        d = desc.get(s, "")
        out.append({"name": f"r/{s}", "description": f"r/{s}. {d}", "rules": f"r/{s}. Rules: {rs}",
                    "rules_desc": f"r/{s}. {d}. Rules: {rs}"}[kind])
    return out


def _pca_norm(C, dz):
    # Compress the raw community embeddings to dz dims via PCA, then L2-normalise each vector so z lives on
    # the unit sphere (keeps the conditioning scale comparable across communities). n_components is capped at
    # len(C)-1 because PCA can't return more components than samples-minus-one.
    z = PCA(n_components=min(dz, len(C) - 1), random_state=SEED).fit_transform(C - C.mean(0))
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def _fold(s, src, te, Y, T, ZB, zmaps, ZPOS, zpos_map, zproto_full, comms):
    # One leave-one-community-out fold: fit on every OTHER community's train rows (`src`), score on s's test
    # rows (`te`). s never contributes to fitting, so its head comes only from its text vector -> cold start.
    X = _Xmm()
    Xsrc = np.asarray(X[src]); ysrc = Y[src].astype(np.float32)
    Xte = np.asarray(X[te]); yte = Y[te]
    o = {"s": s}
    hg = _fit_lr(Xsrc, Y[src]); o["global"] = _auc(Xte @ hg.coef_[0] + hg.intercept_[0], yte)
    Zp = ZB["rules_desc"]

    # 1. RANK x REG sweep, all on the rules+description vector: full grid of capacity x penalty.
    for rank in RANKS:
        for wd in WDS:
            p = _bilinear(Xsrc, ysrc, Zp[src], rank, wd)
            w, b = _wb(p, zmaps["rules_desc"][s]); o[f"rk{rank}_wd{wd:g}"] = _auc(Xte @ w + b, yte)

    # 2. TEXT variants at a fixed mid-capacity config (rank 16, wd 1e-3) so only the text source changes.
    for k in TEXT_KINDS:
        p = _bilinear(Xsrc, ysrc, ZB[k][src], 16, 1e-3)
        w, b = _wb(p, zmaps[k][s]); o[f"text_{k}"] = _auc(Xte @ w + b, yte)

    # 3. WRONG-rules x K: fit once, then score s using its OWN vector (right) vs KWRONG random other
    # communities' vectors (wrong). If the model truly uses z, right should beat wrong; right-minus-wrong ~ 0
    # means the conditioning is inert. Per-draw rng seeded by SEED+kk so the wrong picks are reproducible.
    p8 = _bilinear(Xsrc, ysrc, Zp[src], 8, 1e-3)
    wr, b = _wb(p8, zmaps["rules_desc"][s]); right = _auc(Xte @ wr + b, yte)
    others = [c for c in comms if c != s]
    wrong = []
    for kk in range(KWRONG):
        rng = np.random.default_rng(SEED + kk)
        c = others[int(rng.integers(0, len(others)))]
        ww, bb = _wb(p8, zmaps["rules_desc"][c]); wrong.append(_auc(Xte @ ww + bb, yte))
    o["wr_right"] = right; o["wr_wrong_mean"] = float(np.mean(wrong))
    o["wr_right_minus_wrong"] = right - float(np.mean(wrong))

    # 4. Architecture-free positive control: score directly with the label-derived prototype (removed-minus-kept
    # direction) as the weight vector, no bilinear at all. If THIS separates the classes, per-community signal
    # is expressible as a vector, so any conditioning failure is the architecture's, not a missing signal.
    o["proto_direct"] = _auc(Xte @ zproto_full[s], yte)

    # 5. Positive control through the SAME bilinear, conditioned on the label prototype (rank 64). Should beat
    # global if the architecture can use a community vector at all -- confirms the rules-null isn't just the
    # bilinear being broken.
    pp = _bilinear(Xsrc, ysrc, ZPOS[src], 64, 1e-3)
    w, b = _wb(pp, zpos_map[s]); o["positive_control"] = _auc(Xte @ w + b, yte)
    return o


def run(smoke=False, jobs=8):
    from joblib import Parallel, delayed
    JTMP.mkdir(exist_ok=True)
    from sentence_transformers import SentenceTransformer
    import torch; torch.set_num_threads(2)
    # row index keeps df aligned to the embedding cache; the tox join must be re-sorted by it afterward so
    # row i of df still matches row i of X.
    df = pl.read_parquet(BAL / "slm_mod_split.parquet").select(["subreddit", "idx", "label", "fold"]).with_row_index("row")
    X = np.load(ECACHE, mmap_mode="r"); assert X.shape[0] == df.height
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    df = df.join(tox, on=["subreddit", "idx"], how="left").sort("row")
    sub = df["subreddit"].to_numpy(); Y = df["label"].to_numpy().astype(np.int64)
    # T (toxicity) is carried through but not used in scoring here; null tox defaults to 0.5 (neutral).
    fold = df["fold"].to_numpy(); T = df["tox_toxicity"].fill_null(0.5).to_numpy().astype(np.float64)
    comms = sorted(set(sub.tolist()))
    if smoke:
        comms = comms[:8]
    tr_idx = {s: np.where((sub == s) & (fold == "train"))[0] for s in comms}
    te_idx = {s: np.where((sub == s) & (fold == "test"))[0] for s in comms}
    # Keep only communities with a usable balanced test fold: enough rows and both classes present (AUC is
    # undefined otherwise).
    comms = [s for s in comms if len(te_idx[s]) >= 10 and len(np.unique(Y[te_idx[s]])) == 2]

    enc = SentenceTransformer(ENC_ID, device="cpu")
    ZB, zmaps = {}, {}
    for k in TEXT_KINDS:
        # ENC_PREFIX is e5's required "query: " marker; encode each community's text, PCA+normalise to z,
        # then broadcast z back to every row of that community (Zr) so it can be sliced by src/te masks.
        C = enc.encode([ENC_PREFIX + t for t in _comm_text(comms, k)], batch_size=32,
                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float64)
        z = _pca_norm(C, DZ); zmaps[k] = {s: z[i] for i, s in enumerate(comms)}
        Zr = np.zeros((df.height, z.shape[1]))
        for i, s in enumerate(comms):
            Zr[sub == s] = z[i]
        ZB[k] = Zr

    # Label prototype per community = mean(removed) - mean(kept) over its TRAIN rows: the direction that
    # actually separates the classes, derived from labels. Used as the positive-control conditioning vector.
    Xf = np.asarray(X); protos = []
    for s in comms:
        tr = tr_idx[s]; yy = Y[tr]
        protos.append((Xf[tr][yy == 1].mean(0) - Xf[tr][yy == 0].mean(0)) if (yy == 1).any() and (yy == 0).any()
                      else np.zeros(Xf.shape[1]))
    # zpos: prototypes compressed for the bilinear positive control. zproto_full: raw prototype, scored
    # directly (architecture-free control). Both come from the same label-derived direction.
    zpos = _pca_norm(np.array(protos), 64); zpos_map = {s: zpos[i] for i, s in enumerate(comms)}
    zproto_full = {s: protos[i] for i, s in enumerate(comms)}
    ZPOS = np.zeros((df.height, zpos.shape[1]))
    for i, s in enumerate(comms):
        ZPOS[sub == s] = zpos[i]
    print(f"[stress] {len(comms)} comms, jobs={jobs}, ranks={RANKS}, wds={WDS}", flush=True)

    # src_rows precomputes the leave-one-out training pool for each community (all others' train rows).
    src_rows = {s: np.concatenate([tr_idx[c] for c in comms if c != s]) for s in comms}
    folds = Parallel(n_jobs=jobs, prefer="processes", temp_folder=str(JTMP), max_nbytes="1M")(
        delayed(_fold)(s, src_rows[s], te_idx[s], Y, T, ZB, zmaps, ZPOS, zpos_map, zproto_full, comms) for s in comms)
    F = {f["s"]: f for f in folds}
    rng = np.random.default_rng(SEED)

    # Community-clustered bootstrap (resample communities, not rows) for the across-fold distribution: each
    # community is one observation, so the CI reflects between-community variability. 2000 resamples.
    def med_ci(key):
        v = np.array([F[s][key] for s in comms if F[s].get(key) is not None])
        if not len(v):
            return None
        bt = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
        return {"median": round(float(np.median(v)), 4),
                "ci": [round(float(np.percentile(bt, 2.5)), 4), round(float(np.percentile(bt, 97.5)), 4)]}

    # Paired delta vs the global head, bootstrapped the same way. frac_positive = share of communities where
    # conditioning beats global; the headline test is whether the delta CI clears 0.
    def delta_vs_global(key):
        com = [s for s in comms if F[s].get(key) is not None and F[s].get("global") is not None]
        d = np.array([F[s][key] - F[s]["global"] for s in com])
        bt = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
        return {"median_delta": round(float(np.median(d)), 4),
                "ci": [round(float(np.percentile(bt, 2.5)), 4), round(float(np.percentile(bt, 97.5)), 4)],
                "frac_positive": round(float(np.mean(d > 0)), 3)}

    sweep = {f"rk{r}_wd{w:g}": {"bal_auc": med_ci(f"rk{r}_wd{w:g}"), "delta_vs_global": delta_vs_global(f"rk{r}_wd{w:g}")}
             for r in RANKS for w in WDS}
    # The single most favourable grid cell: if even the best config can't clear global, no cell does.
    best = max(((k, sweep[k]["bal_auc"]["median"]) for k in sweep), key=lambda x: x[1])
    out = {
        "analysis": "coldstart_conditioning_stress_test", "n_communities": len(comms), "epochs": EPOCHS,
        "global_head": med_ci("global"),
        "rank_reg_sweep": sweep,
        "best_sweep_config": {"config": best[0], "bal_auc": best[1],
                              "delta_vs_global": sweep[best[0]]["delta_vs_global"]},
        "text_variants": {k: {"bal_auc": med_ci(f"text_{k}"), "delta_vs_global": delta_vs_global(f"text_{k}")}
                          for k in TEXT_KINDS},
        "wrong_rules_Kdraws": {"K": KWRONG, "right": med_ci("wr_right"), "wrong_mean": med_ci("wr_wrong_mean"),
                               "right_minus_wrong": med_ci("wr_right_minus_wrong")},
        "positive_control_proto_direct": {"bal_auc": med_ci("proto_direct"),
                                          "delta_vs_global": delta_vs_global("proto_direct"),
                                          "note": "ARCHITECTURE-FREE: AUC of X @ (mean train removed - mean train kept). If >> global, per-community signal IS expressible as a community vector (so any conditioning failure is the architecture, not the absence of signal)."},
        "positive_control_labelproto_bilinear": {"bal_auc": med_ci("positive_control"),
                                        "delta_vs_global": delta_vs_global("positive_control"),
                                        "note": "bilinear (rank 64) conditioned on the label prototype; should BEAT global if the bilinear architecture can use a community vector"},
        "verdict_logic": ("If every rank/reg/text config has delta_vs_global <= 0 AND right_minus_wrong ~ 0 AND "
                          "positive_control delta_vs_global >> 0, then conditioning on PUBLIC RULES is genuinely "
                          "inert (the architecture works; rules just carry no transferable signal)."),
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps({"global": out["global_head"], "best_sweep": out["best_sweep_config"],
                      "text_variants": {k: out["text_variants"][k]["delta_vs_global"]["median_delta"] for k in TEXT_KINDS},
                      "right_minus_wrong": out["wrong_rules_Kdraws"]["right_minus_wrong"],
                      "proto_direct_delta": out["positive_control_proto_direct"]["delta_vs_global"],
                      "positive_bilinear_delta": out["positive_control_labelproto_bilinear"]["delta_vs_global"]}, indent=2), flush=True)
    print(f"[stress] WROTE {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); run(smoke=a.smoke, jobs=a.jobs)


if __name__ == "__main__":
    main()
