"""SAE-FEATURE CAUSAL TEST (Test N, NECESSITY) -- does gemma-3-12b-it's moderation decision causally rely on
TOXICITY? Live forward-pass ablation of pre-registered toxicity SAE features vs. a battery of controls.

Per the locked SAE battery spec. Uses google/gemma-scope-2-12b-it JumpReLU resid_post SAEs at L24+L31+L41
(L41 co-ablated to block downstream re-derivation; cross-layer gap directions are near-orthogonal). FROZEN,
pre-registered feature sets are loaded from results/kumar_mod/sae/featsets_L{L}_w16k.json (built by
sae_featid_freeze.py BEFORE any intervention) -- never reselected here.

WHY LIVE (not offline): subtracting decoded contributions from the STORED residual does NOT reduce toxicity
decodability (toxicity is distributed/redundant, re-routes through the unreconstructed residual); only a live
forward-pass hook -- which propagates the edit through all downstream layers and reads the model's own logits --
is valid. The decision-token residual is OOD for the SAE (centered FVU~1) but this is irrelevant: ablation needs
only that w_dec[f] is a real model write-direction, and effects are measured on the model's own logit gap.

ARMS (each comment in the model-removed pool):
  base                              -- clean gap (logit yes - logit no)
  TOX                               -- ablate top-K toxicity features at L24+L31+L41
  DECISION_MATCHED_NONTOX (lynchpin)-- non-toxic features matched on decision-correlation (rules out circularity)
  CLEAN_COMMUNITY                   -- Waller-orthogonal community features (answers "community null is vacuous")
  RANDOM x R (rotating over 20)     -- fire-rate+p95-matched placebo distribution
  dose-response alpha in {0.5,0.6,0.7,0.8,0.9,1.0,1.5}  -- monotone flip-rate = causal signature
  TOX no-L41 ({24,31})              -- under-removes vs {24,31,41} => demonstrates re-derivation mechanism
  TOX decision-token-only           -- separates "reading toxicity" from "deciding on toxicity"
Metrics: decision-flip-to-KEEP rate + mean delta-gap, WITHIN human-label strata, subreddit-clustered bootstrap CI
on each dissociation (tox - control). Causal claim licensed iff flip(TOX) - flip(DECISION_MATCHED_NONTOX) CI95>0.

  smoke: python -m pipeline.kumar_mod.sae_causal_dosedense --smoke
  run:   CUDA_VISIBLE_DEVICES=2 python -m pipeline.kumar_mod.sae_causal_dosedense --n 2000
Out: results/kumar_mod/sae_causal_dosedense.json
"""
from __future__ import annotations
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.decision_axis_collect import _load_model, _build_prompt, MAX_LEN

DEC = ROOT / "results" / "kumar_mod" / "decisiontok"
TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
FSDIR = ROOT / "results" / "kumar_mod" / "sae"
OUT = ROOT / "results" / "kumar_mod" / "sae_causal_dosedense.json"
REPO = "google/gemma-scope-2-12b-it"
# L41 is co-ablated with L24/L31 so the decision can't be re-derived downstream; the no-L41 arm isolates that.
LAYERS = [24, 31, 41]
WIDTH = "16k"
R_RAND = 5          # random placebo draws scored per comment
N_RAND_SETS = 20    # rotating pool of pre-built random feature sets to draw from
SEED = 11           # fixed for the subsample permutation; bootstrap uses SEED+1


def _load_sae(L, dev):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{WIDTH}_l0_medium/params.safetensors")
    sd = load_file(p)
    # Keep SAE params in fp32 even though the model runs lower precision; the encode/decode arithmetic is sensitive.
    return {k: sd[k].to(dev, torch.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")}


def run(n=2000, smoke=False):
    import torch
    meta = pl.read_parquet(DEC / "meta.parquet")
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    # Join on (subreddit, idx); sort by row to keep alignment with the stored decision-token order.
    jj = meta.select(["row", "subreddit", "idx", "label", "gap_rules"]).join(
        tox, on=["subreddit", "idx"], how="left").sort("row")
    # Pre-registered, frozen feature sets (built before any intervention) -- never reselected here.
    FS = {L: json.loads((FSDIR / f"featsets_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}
    rng = np.random.default_rng(SEED)

    tok, model = _load_model()
    dev = next(model.parameters()).device
    layers_mod = (model.model.language_model.layers if hasattr(model.model, "language_model")
                  else model.model.layers)
    tok.truncation_side = "left"
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    SAE = {L: _load_sae(L, dev) for L in LAYERS}

    def encode(x, L):
        # JumpReLU: linear pre-activation gated to zero below the per-feature threshold (no shrinkage above it).
        s = SAE[L]; pre = x @ s["w_enc"] + s["b_enc"]; return pre * (pre > s["threshold"])

    def T(idx_list):
        return torch.tensor(idx_list, device=dev, dtype=torch.long)

    SET = {L: {"tox": T(FS[L]["TOX"]), "decmatch": T(FS[L]["DECISION_MATCHED_NONTOX"]),
               "comm": T(FS[L]["CLEAN_COMMUNITY"]),
               "rand": [T(r) for r in FS[L]["RANDOM"]]} for L in LAYERS}

    def mk(L, F, alpha, dectok):
        wdec = SAE[L]["w_dec"]
        # Live forward hook: subtract the decoded contribution of feature set F from resid_post at this layer,
        # so the edit propagates through all downstream layers (the offline residual-subtraction null does not).
        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32)
            a = encode(hf, L)
            contrib = a[..., F] @ wdec[F]
            if dectok:
                # Restrict the edit to the last (decision) token: tests "deciding on" vs "reading" toxicity.
                c2 = torch.zeros_like(contrib); c2[:, -1, :] = contrib[:, -1, :]; contrib = c2
            # alpha scales the ablation strength; the dose-response sweep relies on this being linear in alpha.
            hf = hf - alpha * contrib
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    def gap(enc, layer_sets=None, alpha=1.0, dectok=False):
        handles = []
        if layer_sets:
            for L, F in layer_sets.items():
                handles.append(layers_mod[L].register_forward_hook(mk(L, F, alpha, dectok)))
        try:
            with torch.no_grad():
                out = model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        # Decision signal = yes/no logit gap at the final position; "yes" means remove under this prompt.
        ll = out.logits[0, -1, :].to(torch.float32)
        return float((ll[yes_id] - ll[no_id]).item())

    # Only comments the model itself flagged for removal under the rules (gap_rules>0): necessity is tested on
    # cases where toxicity could be driving the decision, so an ablation flipping the decision toward KEEP is meaningful.
    pool = jj.filter(pl.col("gap_rules") > 0)
    rows = pool.to_dicts()
    if smoke:
        rows = rows[:10]
    elif n < len(rows):
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
        all3 = {L: SET[L]["tox"] for L in LAYERS}  # ablate toxicity features at all three layers jointly
        rec = {"sub": s, "y": int(r["label"]),
               "base": gap(enc),
               "tox": gap(enc, all3),
               "decmatch": gap(enc, {L: SET[L]["decmatch"] for L in LAYERS}),
               "comm": gap(enc, {L: SET[L]["comm"] for L in LAYERS}),
               # Dense alpha sweep around 1.0: monotone flip-rate growth is the causal dose-response signature.
               "tox_a0.5": gap(enc, all3, alpha=0.5),
               "tox_a0.6": gap(enc, all3, alpha=0.6),
               "tox_a0.7": gap(enc, all3, alpha=0.7),
               "tox_a0.8": gap(enc, all3, alpha=0.8),
               "tox_a0.9": gap(enc, all3, alpha=0.9),
               "tox_a1.5": gap(enc, all3, alpha=1.5),
               # Drop L41: should UNDER-remove vs all-three, evidencing downstream re-derivation of the decision.
               "tox_noL41": gap(enc, {L: SET[L]["tox"] for L in (24, 31)}),
               "tox_dectok": gap(enc, all3, dectok=True)}
        # Rotate through the 20 random sets by (comment index + draw); placebo varies across comments, not fixed.
        for k in range(R_RAND):
            d = (ci + k) % N_RAND_SETS
            rec[f"rand{k}"] = gap(enc, {L: SET[L]["rand"][d] for L in LAYERS})
        recs.append(rec)
        if smoke or (ci + 1) % 200 == 0:
            print(f"[sae N] {ci+1}/{len(rows)} tox={rec['tox']:.2f} base={rec['base']:.2f} "
                  f"decmatch={rec['decmatch']:.2f} comm={rec['comm']:.2f}", flush=True)

    R = pl.DataFrame(recs)
    # dec gates out comments whose clean gap is ~0; a "flip" there is undefined sign noise, so they're excluded.
    gb = R["base"].to_numpy(); dec = np.abs(gb) > 1e-6; y = R["y"].to_numpy()
    rand_cols = [f"rand{k}" for k in range(R_RAND)]

    # Flip rate = fraction of decidable comments whose gap sign changes under the ablation (toward KEEP).
    def flip(col, mask=None):
        g = R[col].to_numpy(); m = dec if mask is None else (dec & mask)
        return float(np.mean((np.sign(g[m]) != np.sign(gb[m])))) if m.sum() else None

    def dgap(col, mask=None):
        g = R[col].to_numpy(); m = dec if mask is None else (dec & mask)
        return float(np.mean((gb - g)[m])) if m.sum() else None

    def rand_flip(mask=None):
        vals = [flip(c, mask) for c in rand_cols]; vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    # Precompute row indices per subreddit so the bootstrap can resample whole clusters.
    subs_u = sorted(set(R["sub"].to_list())); idxby = {s: np.where(R["sub"].to_numpy() == s)[0] for s in subs_u}
    brng = np.random.default_rng(SEED + 1)

    # Subreddit-clustered bootstrap of the dissociation flip(col_a) - flip(col_b): resample subreddits with
    # replacement (not comments) so the CI respects within-community correlation rather than overstating n.
    def boot_diss(col_a, col_b_is_rand=False, col_b=None):
        ga = R[col_a].to_numpy(); rand_arr = [R[c].to_numpy() for c in rand_cols]
        gbb = None if col_b_is_rand else R[col_b].to_numpy(); out = []
        for _ in range(50 if smoke else 2000):
            samp = [subs_u[i] for i in brng.integers(0, len(subs_u), len(subs_u))]
            ix = np.concatenate([idxby[s] for s in samp]); d = dec[ix]
            if d.sum() == 0:
                continue
            fa = np.mean(np.sign(ga[ix][d]) != np.sign(gb[ix][d]))
            if col_b_is_rand:
                # Random control resamples a fresh placebo draw each iteration to fold in placebo variance.
                rc = rand_arr[brng.integers(0, R_RAND)]
                fb = np.mean(np.sign(rc[ix][d]) != np.sign(gb[ix][d]))
            else:
                fb = np.mean(np.sign(gbb[ix][d]) != np.sign(gb[ix][d]))
            out.append(float(fa - fb))
        return out

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)] if a else None

    # Report flip rates within human-label strata so the causal claim must hold regardless of ground-truth label.
    strata = {"all": None, "human_keep": (y == 0), "human_remove": (y == 1)}
    res = {
        "analysis": "sae_feature_causal_toxicity_necessity", "model": "google/gemma-3-12b-it",
        "sae": REPO, "width": WIDTH, "layers": LAYERS, "K_tox": len(FS[31]["TOX"]),
        "n": int(len(R)), "n_subs": len(subs_u), "n_random_draws_per_comment": R_RAND,
        "pool": "model-removed (gap_rules>0)", "tox_features": {f"L{L}": FS[L]["TOX"] for L in LAYERS},
        "flip_rate": {st: {"tox": flip("tox", m), "decmatch": flip("decmatch", m),
                           "community": flip("comm", m), "random": rand_flip(m),
                           "tox_a0.5": flip("tox_a0.5", m), "tox_a1.0": flip("tox", m),
                           "tox_a1.5": flip("tox_a1.5", m), "tox_noL41": flip("tox_noL41", m),
                           "tox_dectok": flip("tox_dectok", m)} for st, m in strata.items()},
        "mean_delta_gap_toward_keep": {"tox": dgap("tox"), "decmatch": dgap("decmatch"),
                                       "community": dgap("comm"),
                                       "random": float(np.mean([dgap(c) for c in rand_cols]))},
        "dissociation_ci95": {
            "tox_minus_decmatch": {"point": round((flip("tox") or 0) - (flip("decmatch") or 0), 4),
                                   "ci95": ci(boot_diss("tox", col_b="decmatch"))},
            "tox_minus_random": {"point": round((flip("tox") or 0) - (rand_flip() or 0), 4),
                                 "ci95": ci(boot_diss("tox", col_b_is_rand=True))},
            "tox_minus_community": {"point": round((flip("tox") or 0) - (flip("comm") or 0), 4),
                                    "ci95": ci(boot_diss("tox", col_b="comm"))}},
        "dose_response_flip": {"a0.5": flip("tox_a0.5"), "a0.6": flip("tox_a0.6"), "a0.7": flip("tox_a0.7"), "a0.8": flip("tox_a0.8"), "a0.9": flip("tox_a0.9"), "a1.0": flip("tox"), "a1.5": flip("tox_a1.5")},
        "re_aggregation": {"tox_L24_31_41": flip("tox"), "tox_L24_31_only": flip("tox_noL41"),
                           "note": "no-L41 should UNDER-remove (decision re-derived downstream)"},
        "interpretation": ("Causal toxicity reliance is licensed iff flip(tox) - flip(decmatch) CI95 excludes 0 "
                           "AND holds within both human-label strata; dose-response monotone in alpha; live hook "
                           "defeats the offline null and the OOD/FVU objection. tox features Neuronpedia-lookupable."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base = OUT if WIDTH == "16k" else OUT.with_name(OUT.stem + f"_{WIDTH}.json")
    out_path = base.with_suffix(".smoke.json") if smoke else base
    out_path.write_text(json.dumps(res, indent=2))
    fr = res["flip_rate"]["all"]; di = res["dissociation_ci95"]
    print(f"[sae N] tox_flip={fr['tox']} decmatch={fr['decmatch']} comm={fr['community']} rand={fr['random']} "
          f"| diss(tox-decmatch)={di['tox_minus_decmatch']['point']} CI={di['tox_minus_decmatch']['ci95']} "
          f"| dose {res['dose_response_flip']} -> {out_path}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--width", default="16k", choices=["16k", "65k"])
    a = ap.parse_args()
    global WIDTH; WIDTH = a.width
    run(n=a.n, smoke=a.smoke)


if __name__ == "__main__":
    main()
