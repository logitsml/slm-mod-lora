"""SAE BATTERY -- Test D: Direct Logit Attribution (DLA).

CPU-only, NO model forward, NO GPU. Decomposes gemma-3-12b-it's moderation logit gap (yes/REMOVE - no/KEEP)
into per-SAE-feature contributions via the logit lens, and asks which features the model READS OUT into the
decision: toxicity vs community.

Math (verified in design): readout = W_U @ FinalNorm(resid). Gemma3RMSNorm gain = (1+weight), eps=1e-6, W_U tied
to embed_tokens. Decision direction d = (1+norm_w) * (emb[yes] - emb[no]). Per comment r=sqrt(mean(x^2)+1e-6).
lens_gap(x) = (x . d)/r = SUM_f acts[f]*(w_dec[f].d)/r + (b_dec.d)/r + (err.d)/r.
Per-feature contribution to REMOVE = acts[:,f] * (w_dec[f].d) / r.  Rank by cov(contrib_f, gap) (discriminative;
NOT signed mean, which an always-on offset feature dominates). Classify top-K by |r(act,tox)| (Detoxify OR ToxiGen)
and Waller community-PC corr. Report the contribution share: toxicity vs community.

  smoke: python -m pipeline.kumar_mod.sae_dla --smoke   (L41 only)
  full:  python -m pipeline.kumar_mod.sae_dla
Out: results/kumar_mod/sae/dla_w{W}.json
"""
from __future__ import annotations
import os
import argparse, glob, json
from pathlib import Path
import numpy as np
import polars as pl
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
DEC = ROOT / "results" / "kumar_mod" / "decisiontok"
TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
MULTITOX = ROOT / "data" / "processed" / "kumar_balanced_multitox.parquet"
WALLER = ROOT / "data" / "external" / "waller_anderson_2021" / "community_embeddings_150d.parquet"
OUTD = ROOT / "results" / "kumar_mod" / "sae"
REPO = "google/gemma-scope-2-12b-it"
MODEL_GLOB = "hf_cache/hub/models--google--gemma-3-12b-it/snapshots/*/model-0000{n}-of-00005.safetensors"


def load_sae(L, width):
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{width}_l0_medium/params.safetensors")
    W = {}
    with safe_open(p, framework="np") as f:
        for k in f.keys():
            W[k] = f.get_tensor(k).astype(np.float32)
    return W


def encode(X, W, chunk=3000):
    acts = np.zeros((X.shape[0], W["w_enc"].shape[1]), dtype=np.float32)
    for i in range(0, X.shape[0], chunk):
        pre = X[i:i + chunk] @ W["w_enc"] + W["b_enc"]
        # JumpReLU gate: keep the pre-activation only where it clears the per-feature threshold (no ReLU).
        acts[i:i + chunk] = pre * (pre > W["threshold"])
    return acts


# Pearson r of every column of A against y in one shot; dead features (zero variance) map to 0, not NaN.
def pcols(A, y):
    af = A - A.mean(0); yf = y - y.mean()
    return np.nan_to_num((af * yf[:, None]).sum(0) / (np.sqrt((af ** 2).sum(0) * (yf ** 2).sum()) + 1e-12))


def decision_dir():
    """d = (1+norm_w) * (emb[yes] - emb[no]); yes/no ids from locus_gate.json (fallback 4443/1904)."""
    # Fallback ids are gemma-3 token ids for the yes/no answer tokens; prefer the captured pair if present.
    yes_id, no_id = 4443, 1904
    lg = DEC / "locus_gate.json"
    if lg.exists():
        g = json.loads(lg.read_text())
        yes_id = int(g.get("yes_id", g.get("yes", yes_id)))
        no_id = int(g.get("no_id", g.get("no", no_id)))
    # embed_tokens lives in shard 1, the final norm gain in shard 5; W_U is tied to embed_tokens.
    shard1 = glob.glob(str(ROOT / MODEL_GLOB.format(n=1)))[0]
    shard5 = glob.glob(str(ROOT / MODEL_GLOB.format(n=5)))[0]
    with safe_open(shard1, framework="pt") as f:
        sl = f.get_slice("language_model.model.embed_tokens.weight")
        ey = sl[yes_id].float().numpy(); en = sl[no_id].float().numpy()
    with safe_open(shard5, framework="pt") as f:
        nw = f.get_tensor("language_model.model.norm.weight").float().numpy()
    # Fold the RMSNorm gain (1+weight) into the unembed difference so d acts directly on the raw residual.
    d = (1.0 + nw) * (ey - en)
    return d.astype(np.float64), yes_id, no_id


def run(width="16k", smoke=False):
    OUTD.mkdir(parents=True, exist_ok=True)
    layers = [41] if smoke else [24, 31, 41]
    d, yes_id, no_id = decision_dir()

    meta = pl.read_parquet(DEC / "meta.parquet")
    # gap_rules is the measured yes-minus-no logit gap captured at decision time (the quantity DLA explains).
    gap = meta["gap_rules"].to_numpy().astype(np.float64)
    subs = meta["subreddit"].to_numpy()
    # Two toxicity scorers: Detoxify (tox_toxicity) and ToxiGen (tox_toxigen), joined back on (subreddit, idx).
    ts = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    mx = pl.read_parquet(MULTITOX).select(["subreddit", "idx", "tox_toxigen"])
    # Sort by row to restore the residual-cache ordering after the joins.
    j = (meta.select(["row", "subreddit", "idx"]).join(ts, on=["subreddit", "idx"], how="left")
         .join(mx, on=["subreddit", "idx"], how="left").sort("row"))
    tox = j["tox_toxicity"].to_numpy().astype(np.float64); tg = j["tox_toxigen"].to_numpy().astype(np.float64)

    uniq = sorted(set(subs.tolist())); sub_to_i = {s: i for i, s in enumerate(uniq)}
    sub_idx = np.array([sub_to_i[s] for s in subs])
    # Waller-Anderson 150d community social-dimension embeddings, keyed case-insensitively; missing -> zero vector.
    wdf = pl.read_parquet(WALLER); wmap = {}
    for r in wdf.iter_rows(named=True):
        key = (r.get("subreddit") or r.get("subreddit_lower") or "").lower()
        if key:
            wmap[key] = r["embedding"]
    Wmat = np.array([wmap.get(s.lower(), [0.0] * 150) for s in uniq], dtype=np.float64)
    # Top-10 community principal components (left singular vectors of the centered embedding matrix).
    U, _, _ = np.linalg.svd(Wmat - Wmat.mean(0), full_matrices=False); commPC = U[:, :10]

    out = {"yes_id": yes_id, "no_id": no_id, "decision_dir_norm": round(float(np.linalg.norm(d)), 4)}
    for L in layers:
        X = np.load(DEC / f"res_rules_L{L}.fp16.npy").astype(np.float32)
        sae = load_sae(L, width)
        acts = encode(X, sae)
        # Per-comment RMS denominator of the final RMSNorm (eps matches Gemma3RMSNorm's 1e-6).
        r = np.sqrt((X.astype(np.float64) ** 2).mean(1) + 1e-6)
        # Each feature's decoder direction projected onto the decision direction: its logit-lens weight.
        wdec_proj = (sae["w_dec"].astype(np.float64) @ d)
        # Per-comment, per-feature contribution to the REMOVE logit: act * (w_dec . d) / RMS.
        contrib = acts.astype(np.float64) * wdec_proj[None, :] / r[:, None]
        feat_sum = contrib.sum(1)
        # Lens gap from the raw residual; cert_corr checks the lens reconstructs the captured gap.
        lens_full = (X.astype(np.float64) @ d) / r
        cert_lens = float(np.corrcoef(lens_full, gap)[0, 1])
        # feat_sum omits bias and reconstruction error, so its agreement with gap bounds the SAE's coverage.
        cert_featsum = float(np.corrcoef(feat_sum, gap)[0, 1])
        sign_agree = float(np.mean(np.sign(lens_full - lens_full.mean()) == np.sign(gap - gap.mean())))


        # Rank by covariance of contribution with the gap, not signed mean: an always-on offset feature
        # has large mean contribution but doesn't track the decision, so cov(contrib, gap) is discriminative.
        cov = (contrib * (gap - gap.mean())[:, None]).mean(0)
        order = np.argsort(-np.abs(cov))

        # A feature is "toxicity" if its activation correlates with either scorer past 0.20 (|r|).
        r_tox = pcols(acts, tox); r_tg = pcols(acts, tg)
        is_tox = (np.abs(r_tox) >= 0.20) | (np.abs(r_tg) >= 0.20)
        # Community signal: per-community mean activation, then max |corr| against any of the 10 community PCs.
        comm_means = np.stack([acts[sub_idx == i].mean(0) for i in range(len(uniq))])
        cm_c = comm_means - comm_means.mean(0); pc_c = commPC - commPC.mean(0)
        wscore = np.abs((cm_c.T @ pc_c) / (np.sqrt((cm_c ** 2).sum(0))[:, None]
                        * np.sqrt((pc_c ** 2).sum(0))[None, :] + 1e-12)).max(1)
        # "community" requires ~is_tox (which already forces |r_tox| < 0.20) plus a clear community signal;
        # the explicit |r_tox| < 0.25 is a redundant guard (looser than 0.20, so subsumed by ~is_tox).
        is_comm = (~is_tox) & (np.abs(r_tox) < 0.25) & (wscore >= 0.30)

        res_L = {"width": width, "cert_corr_lens_gap": round(cert_lens, 3),
                 "cert_corr_featsum_gap": round(cert_featsum, 3), "lens_sign_agreement": round(sign_agree, 3)}
        for K in ([50] if smoke else [20, 50, 100]):
            top = order[:K]
            # Share of the top-K's total |cov| readout that lands on toxicity vs community features.
            tot = np.abs(cov[top]).sum() + 1e-12
            sh_tox = float(np.abs(cov[top][[is_tox[t] for t in top]]).sum() / tot)
            sh_comm = float(np.abs(cov[top][[is_comm[t] for t in top]]).sum() / tot)
            res_L[f"K{K}"] = {
                "share_tox": round(sh_tox, 3), "share_comm": round(sh_comm, 3),
                "share_other": round(1 - sh_tox - sh_comm, 3),
                "frac_tox": round(float(np.mean([is_tox[t] for t in top])), 3),
                "frac_comm": round(float(np.mean([is_comm[t] for t in top])), 3),
                "top_features": [int(t) for t in top[:15]]}
        out[f"L{L}"] = res_L
        print(f"[dla] L{L} w{width}: cert(lens,gap)={cert_lens:.3f} sign_agree={sign_agree:.3f} "
              f"K50 share_tox={res_L['K50']['share_tox']} share_comm={res_L['K50']['share_comm']}", flush=True)
        del acts, contrib

    tag = "_smoke" if smoke else ""
    (OUTD / f"dla_w{width}{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[dla] DONE -> {OUTD}/dla_w{width}{tag}.json", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", default="16k", choices=["16k", "65k"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(width=a.width, smoke=a.smoke)


if __name__ == "__main__":
    main()
