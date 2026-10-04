"""SAE-FEATURE CAUSAL TEST (Test N, NECESSITY) -- does gemma-3-12b-it's moderation decision causally rely on
TOXICITY? Live forward-pass ablation of pre-registered toxicity SAE features vs. a battery of controls.

Per the locked SAE battery spec. Uses google/gemma-scope-2-12b-it JumpReLU resid_post SAEs at L24+L31+L41
(L41 co-ablated to block downstream re-derivation; cross-layer gap directions are near-orthogonal). FROZEN,
pre-registered feature sets are loaded from results/kumar_mod/sae/featsets_L{L}_w{WIDTH}.json (w65k by default for this 65k robustness script; built by
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
  dose-response alpha in {0.5,1.5}  -- monotone flip-rate = causal signature
  TOX no-L41 ({24,31})              -- under-removes vs {24,31,41} => demonstrates re-derivation mechanism
  TOX decision-token-only           -- separates "reading toxicity" from "deciding on toxicity"
Metrics: decision-flip-to-KEEP rate + mean delta-gap, WITHIN human-label strata, subreddit-clustered bootstrap CI
on each dissociation (tox - control). Causal claim licensed iff flip(TOX) - flip(DECISION_MATCHED_NONTOX) CI95>0.

  smoke: python -m pipeline.kumar_mod.sae_causal_toxicity --smoke
  run:   CUDA_VISIBLE_DEVICES=2 python -m pipeline.kumar_mod.sae_causal_toxicity --n 2000
Out: results/kumar_mod/sae_causal_toxicity_65k.json (default width is 65k; this script is the
     65k-width robustness variant. The canonical 16k necessity battery is sae_causal_dosedense.py.)
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
OUT = ROOT / "results" / "kumar_mod" / "sae_causal_toxicity.json"  # filename stem only; output is always width-suffixed (see base= below)
REPO = "google/gemma-scope-2-12b-it"
LAYERS = [24, 31, 41]
WIDTH = "16k"
R_RAND = 5
N_RAND_SETS = 20
SEED = 11


def _load_sae(L, dev):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    # l0_medium variant of the JumpReLU resid_post SAE for this layer; fp32 for stable encode/decode arithmetic.
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{WIDTH}_l0_medium/params.safetensors")
    sd = load_file(p)
    return {k: sd[k].to(dev, torch.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")}


def run(n=2000, smoke=False):
    import torch
    meta = pl.read_parquet(DEC / "meta.parquet")
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    # row order in meta is the decision-token capture order; keep it so hook results line up downstream.
    jj = meta.select(["row", "subreddit", "idx", "label", "gap_rules"]).join(
        tox, on=["subreddit", "idx"], how="left").sort("row")
    # FROZEN pre-registered feature sets (built by sae_featid_freeze.py before any intervention) -- load, never reselect.
    FS = {L: json.loads((FSDIR / f"featsets_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}
    rng = np.random.default_rng(SEED)

    tok, model = _load_model()
    dev = next(model.parameters()).device
    layers_mod = (model.model.language_model.layers if hasattr(model.model, "language_model")
                  else model.model.layers)
    tok.truncation_side = "left"
    # decision is read off the final-token yes/no logit gap; cache the single-token ids once.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    SAE = {L: _load_sae(L, dev) for L in LAYERS}

    # JumpReLU activation: gate each pre-activation against its learned per-feature threshold (hard, not ReLU).
    def encode(x, L):
        s = SAE[L]; pre = x @ s["w_enc"] + s["b_enc"]; return pre * (pre > s["threshold"])

    def T(idx_list):
        return torch.tensor(idx_list, device=dev, dtype=torch.long)

    SET = {L: {"tox": T(FS[L]["TOX"]), "decmatch": T(FS[L]["DECISION_MATCHED_NONTOX"]),
               "comm": T(FS[L]["CLEAN_COMMUNITY"]),
               "rand": [T(r) for r in FS[L]["RANDOM"]]} for L in LAYERS}

    # Build a forward hook that ablates feature set F at layer L by subtracting its decoded write-direction
    # from the live residual. alpha scales the subtraction (dose); dectok restricts the edit to the last token.
    def mk(L, F, alpha, dectok):
        wdec = SAE[L]["w_dec"]
        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32)
            a = encode(hf, L)
            # only the selected features' decoded contribution -- leaves the rest of the reconstruction untouched.
            contrib = a[..., F] @ wdec[F]
            if dectok:
                # zero the edit everywhere but the final (decision) token: isolates "deciding on" from "reading" toxicity.
                c2 = torch.zeros_like(contrib); c2[:, -1, :] = contrib[:, -1, :]; contrib = c2
            hf = hf - alpha * contrib
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    # Live forward pass under the given per-layer ablations, returning the final-token yes-minus-no logit gap.
    # layer_sets=None gives the clean base gap; hooks are always removed afterward so passes stay independent.
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
        ll = out.logits[0, -1, :].to(torch.float32)
        return float((ll[yes_id] - ll[no_id]).item())

    # Only comments the model removed under the real rules (gap_rules>0): ablating toxicity can only flip these to KEEP.
    pool = jj.filter(pl.col("gap_rules") > 0)
    rows = pool.to_dicts()
    if smoke:
        rows = rows[:10]
    elif n < len(rows):
        # seeded subsample for tractability when the pool exceeds n.
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
        all3 = {L: SET[L]["tox"] for L in LAYERS}
        rec = {"sub": s, "y": int(r["label"]),
               "base": gap(enc),                                              # clean gap, no ablation
               "tox": gap(enc, all3),                                         # main arm: toxicity features at L24+L31+L41
               "decmatch": gap(enc, {L: SET[L]["decmatch"] for L in LAYERS}), # lynchpin control: non-tox, decision-correlation matched
               "comm": gap(enc, {L: SET[L]["comm"] for L in LAYERS}),         # Waller-orthogonal community features
               "tox_a0.5": gap(enc, all3, alpha=0.5),                         # dose-response: half / one-and-a-half strength
               "tox_a1.5": gap(enc, all3, alpha=1.5),
               "tox_noL41": gap(enc, {L: SET[L]["tox"] for L in (24, 31)}),   # drop L41 -> decision re-derived downstream, under-removes
               "tox_dectok": gap(enc, all3, dectok=True)}                     # edit final token only
        for k in range(R_RAND):
            # rotate the comment index through the 20 frozen placebo sets so draws decorrelate across comments.
            d = (ci + k) % N_RAND_SETS
            rec[f"rand{k}"] = gap(enc, {L: SET[L]["rand"][d] for L in LAYERS})
        recs.append(rec)
        if smoke or (ci + 1) % 200 == 0:
            print(f"[sae N] {ci+1}/{len(rows)} tox={rec['tox']:.2f} base={rec['base']:.2f} "
                  f"decmatch={rec['decmatch']:.2f} comm={rec['comm']:.2f}", flush=True)

    R = pl.DataFrame(recs)
    gb = R["base"].to_numpy()
    # restrict every metric to comments with a decided base gap; an exactly-zero gap has no sign to flip.
    dec = np.abs(gb) > 1e-6; y = R["y"].to_numpy()
    rand_cols = [f"rand{k}" for k in range(R_RAND)]

    # fraction of decided comments whose gap sign flips under the ablation (decision changed).
    def flip(col, mask=None):
        g = R[col].to_numpy(); m = dec if mask is None else (dec & mask)
        return float(np.mean((np.sign(g[m]) != np.sign(gb[m])))) if m.sum() else None

    # mean shift of the gap toward KEEP (base - ablated); positive => ablation pushed away from REMOVE.
    def dgap(col, mask=None):
        g = R[col].to_numpy(); m = dec if mask is None else (dec & mask)
        return float(np.mean((gb - g)[m])) if m.sum() else None

    # placebo flip rate = mean over the R_RAND random draws.
    def rand_flip(mask=None):
        vals = [flip(c, mask) for c in rand_cols]; vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    # comments the base actually removed (positive gap); flip-to-keep is measured only among these.
    bp = gb > 0

    def to_keep(col):
        return float(np.mean(R[col].to_numpy()[bp] < 0)) if bp.sum() else None

    tk_rand = [v for v in (to_keep(c) for c in rand_cols) if v is not None]

    # precompute per-subreddit row indices for the clustered resample below.
    subs_u = sorted(set(R["sub"].to_list())); idxby = {s: np.where(R["sub"].to_numpy() == s)[0] for s in subs_u}
    brng = np.random.default_rng(SEED + 1)

    # Subreddit-clustered bootstrap on the flip-rate dissociation (tox minus control). Resampling whole
    # subreddits (not rows) respects within-community dependence; col_b_is_rand also redraws a placebo set per rep.
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
                rc = rand_arr[brng.integers(0, R_RAND)]
                fb = np.mean(np.sign(rc[ix][d]) != np.sign(gb[ix][d]))
            else:
                fb = np.mean(np.sign(gbb[ix][d]) != np.sign(gb[ix][d]))
            out.append(float(fa - fb))
        return out

    # percentile 95% CI from the bootstrap replicates.
    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)] if a else None

    # report flip rates within human-label strata so the effect can't be an artifact of one label group.
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
        "base_positive_rate": float(np.mean(bp)),
        "flip_to_keep": {"tox": to_keep("tox"), "decmatch": to_keep("decmatch"),
                         "community": to_keep("comm"),
                         "random": float(np.mean(tk_rand)) if tk_rand else None},
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
        "dose_response_flip": {"a0.5": flip("tox_a0.5"), "a1.0": flip("tox"), "a1.5": flip("tox_a1.5")},
        "re_aggregation": {"tox_L24_31_41": flip("tox"), "tox_L24_31_only": flip("tox_noL41"),
                           "note": "no-L41 should UNDER-remove (decision re-derived downstream)"},
        "interpretation": ("Causal toxicity reliance is licensed iff flip(tox) - flip(decmatch) CI95 excludes 0 "
                           "AND holds within both human-label strata; dose-response monotone in alpha; live hook "
                           "defeats the offline null and the OOD/FVU objection. tox features Neuronpedia-lookupable."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base = OUT.with_name(OUT.stem + f"_{WIDTH}.json")  # always width-suffixed: sae_causal_toxicity_65k.json (default) or _16k.json, so widths never clobber each other
    # 16k width writes the canonical filename; other widths and smoke runs get suffixed so they never clobber it.
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
    ap.add_argument("--width", default="65k", choices=["16k", "65k"])  # 65k is this script's role; the 16k necessity battery is sae_causal_dosedense.py
    a = ap.parse_args()
    global WIDTH; WIDTH = a.width
    run(n=a.n, smoke=a.smoke)


if __name__ == "__main__":
    main()
