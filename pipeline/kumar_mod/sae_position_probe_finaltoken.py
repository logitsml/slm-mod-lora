"""Final-token-only SAE necessity probe (GPU; the final-token arm).

The position probe (sae_position_probe.py) established: ablating the frozen
toxicity features at every position flips 100% of model removals, the
post-comment decision segment alone flips 96.35%, and the comment body alone
flips 0.25%. This script measures the final-token-only arm (55.6%): identical frozen features
(featsets_L{24,31,41}_w16k), identical live subtract hook, identical seed-11
sample of 2,000 model-removed comments, with the position mask restricted to
the last sequence position (the token whose residual stream produces the
yes/no logits). The "all" arm is re-run as an internal anchor and must
reproduce 1.0.

  run: CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.sae_position_probe_finaltoken --n 2000
Out: results/kumar_mod/sae_position_probe_finaltoken.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from pipeline.kumar_mod._common import RES, ROOT, SEED
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.decision_axis_collect import MAX_LEN, _build_prompt, _load_model
from pipeline.kumar_mod.sae_position_probe import _load_sae, LAYERS, REPO, WIDTH

DEC = RES / "decisiontok"
FSDIR = RES / "sae"
OUT = RES / "sae_position_probe_finaltoken.json"


def run(n=2000, smoke=False):
    import torch
    meta = pl.read_parquet(DEC / "meta.parquet")
    jj = meta.select(["row", "subreddit", "idx", "label", "gap_rules"]).sort("row")
    FS = {L: json.loads((FSDIR / f"featsets_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}
    rng = np.random.default_rng(SEED)

    tok, model = _load_model()
    dev = next(model.parameters()).device
    layers_mod = (model.model.language_model.layers if hasattr(model.model, "language_model")
                  else model.model.layers)
    # Left-truncate: keep the tail of long prompts so the comment + decision scaffold (the readout site)
    # always survives, rather than dropping it from the front.
    tok.truncation_side = "left"
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    SAE = {L: _load_sae(L, dev) for L in LAYERS}
    # Frozen consensus toxicity feature ids per layer; these come from featsets_L*.json, identical to the
    # set used by sae_position_probe -- nothing re-fit here.
    TOXSET = {L: torch.tensor(FS[L]["TOX"], device=dev, dtype=torch.long) for L in LAYERS}

    def encode(x, L):
        # JumpReLU encode: linear pre-activation gated by the per-feature threshold (hard gate, not ReLU).
        s = SAE[L]
        pre = x @ s["w_enc"] + s["b_enc"]
        return pre * (pre > s["threshold"])

    def mk(L, F, pos):
        wdec = SAE[L]["w_dec"]

        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32)
            a = encode(hf, L)
            # Subtract only the toxicity features' decoder contribution from the residual stream -- a live
            # surgical edit, leaving every other feature's reconstruction untouched.
            contrib = a[..., F] @ wdec[F]
            if pos is not None:
                # Restrict the edit to the given position(s): zero the contribution everywhere else so
                # ablation acts on one segment of the sequence at a time.
                pm = torch.zeros(hf.shape[1], device=hf.device, dtype=hf.dtype)
                pm[pos] = 1.0
                contrib = contrib * pm[None, :, None]
            hf = hf - contrib
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    def gap(enc, pos="NONE"):
        handles = []
        if pos != "NONE":
            p = None if pos == "ALL" else pos
            for L in LAYERS:
                handles.append(layers_mod[L].register_forward_hook(mk(L, TOXSET[L], p)))
        try:
            with torch.no_grad():
                out = model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        # Decision signal = yes/no logit gap at the last position; sign flip vs base = the model's
        # remove/keep call changed.
        ll = out.logits[0, -1, :].to(torch.float32)
        return float((ll[yes_id] - ll[no_id]).item())

    # Only comments the model actually removed under its own rules (positive base gap); these are the cases
    # whose decision an ablation can flip.
    pool = jj.filter(pl.col("gap_rules") > 0)
    rows = pool.to_dicts()
    if smoke:
        rows = rows[:10]
    elif n < len(rows):
        # Seed-11 permutation draws the SAME 2,000-comment subsample as sae_position_probe, so the
        # final-token arm is measured on identical inputs to the all-position anchor.
        rows = [rows[i] for i in rng.permutation(len(rows))[:n]]

    bodies, recs = {}, []
    for ci, r in enumerate(rows):
        s = r["subreddit"]
        if s not in bodies:
            bodies[s] = K.load_comments(s)
        body = bodies[s][r["idx"]][0]
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        enc = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        seqlen = enc["input_ids"].shape[1]
        # The final token: the one position whose residual stream produces the yes/no logits. The
        # final-token arm ablates only here.
        last = torch.tensor([seqlen - 1], device=dev, dtype=torch.long)
        # base = clean (no hook); all = ablate every position (anchor, must reproduce 1.0); final_token =
        # ablate the last position only.
        rec = {"sub": s, "base": gap(enc), "all": gap(enc, "ALL"), "final_token": gap(enc, last)}
        recs.append(rec)
        if smoke or (ci + 1) % 200 == 0:
            print(f"[ft] {ci+1}/{len(rows)} base={rec['base']:.1f} all={rec['all']:.1f} "
                  f"final={rec['final_token']:.1f}", flush=True)

    R = pl.DataFrame(recs)
    gb = R["base"].to_numpy()
    # Drop near-zero base gaps: an undecided comment has no decision to flip, so its sign is meaningless.
    dec = np.abs(gb) > 1e-6
    subs_u = sorted(set(R["sub"].to_list()))
    # Row indices grouped by subreddit, precomputed once for the clustered bootstrap resample below.
    idxby = {s: np.where(R["sub"].to_numpy() == s)[0] for s in subs_u}
    # SEED+1 (not SEED) so the bootstrap stream is independent of the sampling RNG above.
    brng = np.random.default_rng(SEED + 1)

    def flip(col):
        # Fraction of decided comments whose ablated logit-gap sign disagrees with base = decisions flipped.
        g = R[col].to_numpy()
        return float(np.mean(np.sign(g[dec]) != np.sign(gb[dec])))

    def boot_ci(col):
        g = R[col].to_numpy()
        out = []
        for _ in range(50 if smoke else 2000):
            # Subreddit-clustered bootstrap: resample whole communities with replacement (not individual
            # comments), so the CI reflects between-community variance rather than treating correlated
            # within-community rows as independent.
            samp = [subs_u[i] for i in brng.integers(0, len(subs_u), len(subs_u))]
            ix = np.concatenate([idxby[s] for s in samp])
            d = dec[ix]
            if d.sum():
                out.append(float(np.mean(np.sign(g[ix][d]) != np.sign(gb[ix][d]))))
        return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]

    res = {"analysis": "sae_necessity_final_token_only", "model": "google/gemma-3-12b-it",
           "sae": REPO, "width": WIDTH, "layers": LAYERS, "n": int(len(R)),
           "n_subs": len(subs_u), "pool": "model-removed (gap_rules>0), same seed-11 sample as sae_position_probe",
           "flip_rate": {"all": flip("all"), "final_token": flip("final_token")},
           "flip_ci95": {"all": boot_ci("all"), "final_token": boot_ci("final_token")},
           "paper_claim": "final decision token alone flips 56% of removals",
           "anchor": "all-position arm must reproduce 1.0 (sae_position_probe.json)"}
    out_path = OUT.with_suffix(".smoke.json") if smoke else OUT
    out_path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["flip_rate"], indent=2), "->", out_path, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    run(n=a.n, smoke=a.smoke)


if __name__ == "__main__":
    main()
