"""SAE BATTERY -- EXTRA CONTROLS for the causal necessity test (gemma-3-12b-it).

Adds five control feature sets, FROZEN before any GPU intervention, reusing the EXACT
per-feature statistics and matching helpers from sae_featid_freeze (identical fire-rate, p95, r_tox, r_gap,
partial-r definitions) and the SAME locked TOX set, so the controls are directly comparable.

Controls (per layer L in {24,31,41}; K=10; all NON-toxic |r_tox|<0.10 except PERMLABEL and HELDOUT_TOX):
  ACTFREQ_MATCHED : non-tox, matched to each TOX feature on firing-rate (+p95) +-20%   -> "not just busy features"
  DECNORM_MATCHED : non-tox, matched to each TOX feature on ||w_dec|| +-20%            -> "not just big residual edits"
  DLA_MATCHED     : non-tox, matched on |w_dec . U|, U = final-norm-folded (E[yes]-E[no]) -> "not just decision-readout size"
  PERMLABEL       : TOX selection re-run on PERMUTED toxicity labels (one shared perm)  -> null: should NOT flip
  HELDOUT_TOX     : TOX selection on comment-half A; eval on disjoint half B            -> no selection-on-eval circularity

Out: results/kumar_mod/sae/controls_extra_L{L}_w16k.json  and  controls_heldout_split.json
CPU only. No model load except a one-time read of the embedding + final-norm tensors for the DLA direction.

  smoke (L31 only): python -m pipeline.kumar_mod.sae_controls_freeze --smoke
  full:             python -m pipeline.kumar_mod.sae_controls_freeze
"""
from __future__ import annotations
import os
import argparse, json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
import sys; sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod.sae_featid_freeze import (
    DEC, TOXPQ, MULTITOX, OUTD, REPO, K_HEAD, SEED, load_sae, encode, pcols, partial_cols, match_pool)

MODEL = "google/gemma-3-12b-it"
WIDTH = "16k"
FRAC = 0.20


def load_unembed_dir(yes_id, no_id):
    """Final-norm-folded logit-lens decision direction U = (1+norm_w) ⊙ (E[yes]-E[no]); gemma ties embeddings."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    widx = json.loads(Path(hf_hub_download(MODEL, "model.safetensors.index.json")).read_text())["weight_map"]
    def find(suffix):
        ks = [k for k in widx if k.endswith(suffix)]
        assert len(ks) == 1, f"expected 1 tensor ending {suffix}, got {ks}"
        return ks[0]
    def get(name):
        with safe_open(hf_hub_download(MODEL, widx[name]), framework="pt") as f:
            return f.get_tensor(name).float().numpy()
    # Gemma ties unembed to the input embedding, so the yes/no readout row is E[yes]-E[no].
    E = get(find("embed_tokens.weight"))
    nw = get(find("model.norm.weight")).astype(np.float32)
    u = E[yes_id].astype(np.float32) - E[no_id].astype(np.float32)
    # Fold the final RMSNorm gain (gemma stores it as 1+nw) so U acts on raw residual-stream vectors.
    return (1.0 + nw) * u


def match_scalar(tox_idx, cand, stat, frac=FRAC, used=None):
    """Greedy nearest non-replacement match of each TOX feature to a candidate on a scalar stat (|.|)."""
    used = set() if used is None else set(used)
    picks, st = [], np.abs(stat)
    cand = [int(c) for c in cand]
    for t in tox_idx:
        target = st[t]
        # Prefer candidates within +-frac of the TOX feature's stat; fall back to any unused if the band is empty.
        pool = [c for c in cand if c not in used and abs(st[c] - target) <= frac * abs(target) + 1e-9]
        if not pool:
            pool = [c for c in cand if c not in used]
        if not pool:
            break
        c = min(pool, key=lambda c: abs(st[c] - target))
        picks.append(int(c)); used.add(c)
    return picks


def _stats(acts, tox, tg, snlp, vneg, gap_v, rows=None):
    """Per-feature stats on a (sub)set of rows; identical formulas to sae_featid_freeze."""
    A = acts if rows is None else acts[rows]
    t, g, s, v, gp = (x if rows is None else x[rows] for x in (tox, tg, snlp, vneg, gap_v))
    fire = (A > 0).mean(0)
    # p95 of the active-only distribution (conditional on firing), not over all rows including zeros.
    p95 = np.zeros(A.shape[1])
    for f in range(A.shape[1]):
        on = A[:, f][A[:, f] > 0]
        if on.size:
            p95[f] = np.percentile(on, 95)
    # pc_tox_v is the toxicity correlation partialled on VADER negativity: separates "tracks toxicity" from "tracks negativity".
    return dict(fire=fire, p95=p95, r_tox=pcols(A, t), r_tg=pcols(A, g),
                r_snlp=pcols(A, s), r_gap=pcols(A, gp), pc_tox_v=partial_cols(A, t, v))


def _select_tox(st, fire):
    """Replicate the locked TOX selection from given stats: consensus |r|>=.20 (>=2 of 3) gate, rank pc_tox|vneg."""
    firing = fire > 1e-6
    # Consensus gate: at least 2 of 3 independent toxicity scorers (Detoxify, ToxiGen, s-nlp) must agree |r|>=.20,
    # so no single classifier's idiosyncrasy can put a feature in TOX.
    consensus = ((np.abs(st["r_tox"]) >= 0.20).astype(int) + (np.abs(st["r_tg"]) >= 0.20).astype(int)
                 + (np.abs(st["r_snlp"]) >= 0.20).astype(int)) >= 2
    elig = firing & consensus
    # Among eligible features, rank by sentiment-controlled toxicity correlation; non-eligible pushed to -inf.
    return np.argsort(-np.where(elig, st["pc_tox_v"], -np.inf))[:K_HEAD]


def run(width="16k", smoke=False):
    layers = [31] if smoke else [24, 31, 41]
    meta = pl.read_parquet(DEC / "meta.parquet")
    n = len(meta)
    gap_v = meta["gap_rules"].to_numpy().astype(np.float64)
    subs = meta["subreddit"].to_numpy()
    idxs = meta["idx"].to_numpy()
    ts = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity", "vader_neg"])
    mx = pl.read_parquet(MULTITOX).select(["subreddit", "idx", "tox_toxigen", "tox_snlp"])
    # Join the three scorer tables onto meta by (subreddit, idx), then re-sort by row so arrays line up with the
    # activation matrix row order. The length assert guards against a fan-out from duplicate keys.
    j = (meta.select(["row", "subreddit", "idx"]).join(ts, on=["subreddit", "idx"], how="left")
         .join(mx, on=["subreddit", "idx"], how="left").sort("row"))
    assert len(j) == n, "join changed row count"
    tox = j["tox_toxicity"].to_numpy().astype(np.float64)
    tg = j["tox_toxigen"].to_numpy().astype(np.float64)
    snlp = j["tox_snlp"].to_numpy().astype(np.float64)
    vneg = j["vader_neg"].to_numpy().astype(np.float64)


    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL)
    yes_id = tk.encode("yes", add_special_tokens=False)[0]
    no_id = tk.encode("no", add_special_tokens=False)[0]
    U = load_unembed_dir(yes_id, no_id)
    print(f"[ctrl] decision dir U dim={U.shape} (yes={yes_id} no={no_id})", flush=True)


    # Seed 11 is the project-wide split seed (shared with the decision/encoder arms) so halves are reproducible.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    # Half A selects HELDOUT_TOX; half B is the disjoint comment set the GPU intervention later evaluates on,
    # so feature selection never sees the eval rows (no selection-on-eval circularity).
    half = perm[: n // 2]; otherhalf = perm[n // 2:]
    A_mask = np.zeros(n, bool); A_mask[half] = True
    heldout_B = [{"subreddit": str(subs[i]), "idx": int(idxs[i])} for i in otherhalf]


    # Separate stream (SEED+42) for the label-permutation null, independent of the A/B split permutation.
    permL = np.random.default_rng(SEED + 42).permutation(n)

    out_summary = {}
    for L in layers:
        X = np.load(DEC / f"res_rules_L{L}.fp16.npy").astype(np.float32)
        W = load_sae(L, width)
        acts = encode(X, W)
        wdec = W["w_dec"]
        assert wdec.shape[0] == acts.shape[1], "w_dec/feature axis mismatch"
        # ||w_dec|| = how big a residual edit each feature makes; dla = its projection onto the yes/no readout U
        # (direct-logit-attribution), i.e. how much each feature moves the decision.
        dec_norm = np.linalg.norm(wdec, axis=1)
        dla = wdec @ U.astype(np.float32)

        st = _stats(acts, tox, tg, snlp, vneg, gap_v)
        fire = st["fire"]
        # Re-select TOX here from the same stats/gate as the locked set; cross-checked against featsets_* below.
        TOX = [int(x) for x in _select_tox(st, fire)]

        # Control candidate pool: features that fire but are toxicity-null (|r_tox|<0.10), excluding TOX itself.
        firing = fire > 1e-6
        nontox = np.where(firing & (np.abs(st["r_tox"]) < 0.10))[0]
        nontox = np.array([c for c in nontox if c not in set(TOX)])

        # ACTFREQ: one non-tox match per TOX feature on (fire-rate, p95) within +-20%, no replacement,
        # tie-broken by closest fire-rate. Controls for "it's just a busy feature".
        ACTFREQ = []
        used = set()
        for t in TOX:
            pool = np.array([c for c in nontox if c not in used])
            m = match_pool(pool, t, fire, st["p95"]) if len(pool) else pool
            if len(m) == 0:
                m = pool
            if len(m) == 0:
                break
            c = int(min(m, key=lambda c: abs(fire[c] - fire[t])))
            ACTFREQ.append(c); used.add(c)

        # Scalar-matched controls: same nontox pool matched on decoder-norm and on |DLA| respectively.
        DECNORM = match_scalar(TOX, nontox, dec_norm)
        DLAM = match_scalar(TOX, nontox, dla)


        # PERMLABEL null: shuffle all four scorer vectors by the SAME perm (keeps their joint structure) and re-run
        # selection. If TOX is real, the consensus gate should not produce a toxicity-correlated set under permuted labels.
        stP = _stats(acts, tox[permL], tg[permL], snlp[permL], vneg[permL], gap_v)
        PERM = [int(x) for x in _select_tox(stP, fire)]


        # HELDOUT_TOX: re-derive stats and select on half A only; this set is later evaluated on disjoint half B.
        stA = _stats(acts, tox, tg, snlp, vneg, gap_v, rows=half)
        HELD = [int(x) for x in _select_tox(stA, stA["fire"])]

        ctrl = {
            "layer": L, "width": width, "K": K_HEAD, "seed": SEED,
            "TOX_reselected_here": TOX,
            "ACTFREQ_MATCHED": ACTFREQ, "DECNORM_MATCHED": DECNORM, "DLA_MATCHED": DLAM,
            "PERMLABEL": PERM, "HELDOUT_TOX": HELD,
            "match_quality": {
                "fire_tox_mean": round(float(fire[TOX].mean()), 4),
                "fire_actfreq_mean": round(float(fire[ACTFREQ].mean()), 4) if ACTFREQ else None,
                "decnorm_tox_mean": round(float(dec_norm[TOX].mean()), 4),
                "decnorm_match_mean": round(float(dec_norm[DECNORM].mean()), 4) if DECNORM else None,
                "absdla_tox_mean": round(float(np.abs(dla[TOX]).mean()), 4),
                "absdla_match_mean": round(float(np.abs(dla[DLAM]).mean()), 4) if DLAM else None,
                "rtox_actfreq_mean": round(float(np.abs(st["r_tox"][ACTFREQ]).mean()), 4) if ACTFREQ else None,
                "rtox_decnorm_mean": round(float(np.abs(st["r_tox"][DECNORM]).mean()), 4) if DECNORM else None,
                "rtox_dla_mean": round(float(np.abs(st["r_tox"][DLAM]).mean()), 4) if DLAM else None,
                "rtox_perm_mean": round(float(np.abs(st["r_tox"][PERM]).mean()), 4) if PERM else None,
                "rtox_held_mean": round(float(np.abs(st["r_tox"][HELD]).mean()), 4) if HELD else None,
                "rgap_tox_mean": round(float(np.abs(st["r_gap"][TOX]).mean()), 4),
                "rgap_perm_mean": round(float(np.abs(st["r_gap"][PERM]).mean()), 4) if PERM else None,
                "rgap_actfreq_mean": round(float(np.abs(st["r_gap"][ACTFREQ]).mean()), 4) if ACTFREQ else None,
                "rgap_dla_mean": round(float(np.abs(st["r_gap"][DLAM]).mean()), 4) if DLAM else None,
                "held_vs_tox_overlap": len(set(HELD) & set(TOX)),
            },
        }
        (OUTD / f"controls_extra_L{L}_w{width}.json").write_text(json.dumps(ctrl, indent=2))
        out_summary[f"L{L}"] = ctrl["match_quality"] | {"TOX": TOX, "held_overlap": len(set(HELD) & set(TOX))}
        mq = ctrl["match_quality"]
        print(f"[ctrl] L{L}: TOX={TOX[:4]}... | actfreq fire {mq['fire_tox_mean']}vs{mq['fire_actfreq_mean']} "
              f"| decnorm {mq['decnorm_tox_mean']}vs{mq['decnorm_match_mean']} "
              f"| |dla| {mq['absdla_tox_mean']}vs{mq['absdla_match_mean']} "
              f"| ctrl|r_tox|~{mq['rtox_dla_mean']} | perm|r_tox|~{mq['rtox_perm_mean']} "
              f"| held∩tox={mq['held_vs_tox_overlap']}", flush=True)
        del acts


    # Sanity check: the TOX set re-derived here must exactly equal the locked set from sae_featid_freeze,
    # confirming the controls are built against the same frozen features the GPU intervention will ablate.
    for L in layers:
        locked = json.loads((OUTD / f"featsets_L{L}_w{width}.json").read_text())["TOX"]
        here = out_summary[f"L{L}"]["TOX"]
        print(f"[ctrl] L{L} TOX matches locked featsets: {set(locked) == set(here)} "
              f"(locked {sorted(locked)[:4]}... here {sorted(here)[:4]}...)", flush=True)

    (OUTD / "controls_heldout_split.json").write_text(json.dumps(
        {"seed": SEED, "n": n, "n_heldout_B": len(heldout_B), "heldout_B": heldout_B}, indent=2))
    print(f"[ctrl] DONE -> {OUTD}/controls_extra_L*_w{width}.json (+ heldout split, {len(heldout_B)} B-comments)", flush=True)
    return out_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", default="16k", choices=["16k", "65k"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global WIDTH; WIDTH = a.width
    run(width=a.width, smoke=a.smoke)


if __name__ == "__main__":
    main()
