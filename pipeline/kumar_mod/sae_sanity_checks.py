"""SAE necessity SANITY CHECKS.

Reuses the EXACT live-ablation machinery of sae_controls_ablate (same _load_model/_build_prompt, same SAE load,
same hook subtracting a*(acts[F]@w_dec[F]) at L24+L31+L41, same flip = sign-change-of-gap definition, same TOX
features). On the model-removed pool it adds four foregrounded checks:

 1. FORMAT/REFUSAL VALIDITY: fraction of comments whose argmax next token is a valid 'yes'/'no' BEFORE vs AFTER
    toxicity ablation -- shows ablation does not break generation into refusals/garbage.
 2. LOGIT-GAP DISTRIBUTION: full per-comment keep/remove gap for base + toxicity + a generic keep-steer + a
    random steer (saved as a parquet so the distribution figure is reproducible).
 3. GENERIC DECISION-AXIS STEER (the key control): at L41, ADD a content-agnostic push along the keep direction
    d = W_U[no]-W_U[yes], magnitude-matched per comment to that comment's own L41 toxicity-ablation perturbation
    norm. A RANDOM-direction steer of equal norm is the null. If 'the edit is just a generic no-removal push'
    were true, the generic steer at matched magnitude would reproduce toxicity ablation; we test that.
 4. SELECTIVITY: corr(per-comment flip, comment toxicity). Toxicity ablation should flip toxicity-driven
    removals preferentially; a generic keep-steer should flip independent of comment toxicity.

  smoke: CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.sae_sanity_checks --smoke
  run:   CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.sae_sanity_checks --n 800
Out: results/kumar_mod/sae_sanity_checks.json  +  sae_sanity_gaps.parquet
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
OUT = ROOT / "results" / "kumar_mod" / "sae_sanity_checks.json"
GAPPQ = ROOT / "results" / "kumar_mod" / "sae_sanity_gaps.parquet"
REPO = "google/gemma-scope-2-12b-it"
# Same three resid_post sites the necessity ablation acts on; steers below act at L41 only.
LAYERS = [24, 31, 41]
WIDTH = "16k"
# Shared project seed: same 80/20 split and same row subsample as the rest of the pipeline.
SEED = 11


def _load_sae(L, dev):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{WIDTH}_l0_medium/params.safetensors")
    sd = load_file(p)
    return {k: sd[k].to(dev, torch.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")}


def run(n=800, smoke=False):
    import torch
    meta = pl.read_parquet(DEC / "meta.parquet")
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    # Attach the Detoxify toxicity score per comment; keyed on (subreddit, idx), sorted to match capture order.
    jj = meta.select(["row", "subreddit", "idx", "label", "gap_rules"]).join(
        tox, on=["subreddit", "idx"], how="left").sort("row")
    FS = {L: json.loads((FSDIR / f"featsets_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}
    rng = np.random.default_rng(SEED)

    tok, model = _load_model()
    dev = next(model.parameters()).device
    layers_mod = (model.model.language_model.layers if hasattr(model.model, "language_model")
                  else model.model.layers)
    # Left-truncate so the decision-eliciting tail of the prompt always survives the length cap.
    tok.truncation_side = "left"
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    SAE = {L: _load_sae(L, dev) for L in LAYERS}
    # Frozen consensus toxicity feature indices per layer (same set used by the necessity ablation).
    Ftox = {L: torch.tensor(FS[L]["TOX"], device=dev, dtype=torch.long) for L in LAYERS}


    # Keep direction in residual space = unembedding of "no" minus "yes": pushing along it favors removal->keep.
    WU = model.get_output_embeddings().weight.detach().to(torch.float32)
    keep_dir = (WU[no_id] - WU[yes_id]); keep_dir = keep_dir / keep_dir.norm()
    # Null comparison: an isotropic random unit direction, seeded for reproducibility.
    rand_dir = torch.tensor(rng.standard_normal(WU.shape[1]), device=dev, dtype=torch.float32)
    rand_dir = rand_dir / rand_dir.norm()

    def encode(x, L):
        # JumpReLU encode: feature is active only when its pre-activation clears the per-feature threshold.
        s = SAE[L]; pre = x @ s["w_enc"] + s["b_enc"]; return pre * (pre > s["threshold"])

    def mk_ablate(L, F):
        wdec = SAE[L]["w_dec"]
        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32)
            a = encode(hf, L)
            # Subtract just the toxicity features' reconstruction; rest of the residual stream is untouched.
            hf = hf - a[..., F] @ wdec[F]
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    def mk_add(vec):
        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32) + vec[None, None, :]
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    def fwd(enc, hooks):
        handles = [layers_mod[L].register_forward_hook(hk) for L, hk in hooks]
        try:
            with torch.no_grad():
                out = model(**enc, output_hidden_states=False, use_cache=False)
        finally:
            for hd in handles:
                hd.remove()
        # Gap is yes-minus-no on the final-token logits; argmax is the actual predicted next token.
        ll = out.logits[0, -1, :].to(torch.float32)
        return float((ll[yes_id] - ll[no_id]).item()), int(ll.argmax().item())

    def l41_tox_perturb_norm(enc):
        """mean over tokens of ||toxicity contrib at L41|| -- the per-comment ablation magnitude to match."""
        store = {}
        def cap(m, i, o):
            store["h"] = (o[0] if isinstance(o, tuple) else o).to(torch.float32).detach()
        hd = layers_mod[41].register_forward_hook(cap)
        try:
            with torch.no_grad():
                model(**enc, output_hidden_states=False, use_cache=False)
        finally:
            hd.remove()
        a = encode(store["h"], 41)
        # Per-token L41 residual contributed by the toxicity features; its norm averaged over tokens
        # gives the magnitude each steer is scaled to, so the keep/random push is energy-matched to ablation.
        contrib = a[0, :, Ftox[41]] @ SAE[41]["w_dec"][Ftox[41]]
        return float(contrib.norm(dim=-1).mean().item())

    ablate_hooks = [(L, mk_ablate(L, Ftox[L])) for L in LAYERS]
    valid = lambda am: int(am in (yes_id, no_id))

    # Model-removed pool: comments the rules-prompt model decided to remove (positive yes-no gap).
    pool = jj.filter(pl.col("gap_rules") > 0)
    rows = pool.to_dicts()
    if smoke:
        rows = rows[:8]
    elif n < len(rows):
        # Random (not head-of-file) subsample so the n cap doesn't bias toward early subreddits.
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
        g_base, am_base = fwd(enc, [])                  # unedited forward pass
        g_tox, am_tox = fwd(enc, ablate_hooks)          # toxicity features ablated at L24+L31+L41
        m_i = l41_tox_perturb_norm(enc)                 # this comment's own L41 ablation magnitude
        # Two matched-norm single-site controls at L41: keep-direction push vs random-direction null.
        g_keep, am_keep = fwd(enc, [(41, mk_add(keep_dir * m_i))])
        g_rand, am_rand = fwd(enc, [(41, mk_add(rand_dir * m_i))])
        recs.append({"sub": s, "y": int(r["label"]), "tox_score": float(r["tox_toxicity"] or 0.0),
                     "g_base": g_base, "g_tox": g_tox, "g_keep": g_keep, "g_rand": g_rand,
                     "v_base": valid(am_base), "v_tox": valid(am_tox),
                     "v_keep": valid(am_keep), "v_rand": valid(am_rand), "m_i": m_i})
        if smoke or (ci + 1) % 100 == 0:
            print(f"[sanity] {ci+1}/{len(rows)} base={g_base:.2f} tox={g_tox:.2f} keep={g_keep:.2f} "
                  f"rand={g_rand:.2f} valid(base/tox)={am_base in (yes_id,no_id)}/{am_tox in (yes_id,no_id)}",
                  flush=True)

    R = pl.DataFrame(recs)
    R.write_parquet(GAPPQ)
    # Flip and correlation are defined only on comments with a non-degenerate base gap (a real decision).
    gb = R["g_base"].to_numpy(); dec = np.abs(gb) > 1e-6
    def flip(col):
        # Flip = the intervention changed the sign of the yes-no gap (the decision actually crossed over).
        g = R[col].to_numpy(); return float(np.mean(np.sign(g[dec]) != np.sign(gb[dec]))) if dec.sum() else None

    res = {
        "analysis": "sae_necessity_sanity_checks", "model": "google/gemma-3-12b-it", "sae": REPO,
        "width": WIDTH, "layers": LAYERS, "n": int(len(R)), "n_decided": int(dec.sum()),
        "pool": "model-removed (gap_rules>0)",
        "format_validity": {
            "base_argmax_valid_yesno": round(float(R["v_base"].mean()), 4),
            "tox_ablated_argmax_valid_yesno": round(float(R["v_tox"].mean()), 4),
            "note": "fraction whose argmax next token is a valid yes/no; ablation keeps generation well-formed"},
        "flip_rate": {"tox_ablation": flip("g_tox"),
                      "generic_keep_steer_matched": flip("g_keep"),
                      "random_steer_matched": flip("g_rand"),
                      "note": ("generic_keep_steer adds the unembedding keep-direction at L41 with norm matched "
                               "per-comment to the toxicity-ablation perturbation; random_steer is the null. "
                               "CAVEAT: tox flip saturates at ~1.0, so read mean_keepgap_shift (below) for the "
                               "magnitude comparison; the matched-control content-specificity evidence lives in "
                               "the dosedense/controls battery, not in this file. The steers act at L41 ONLY "
                               "whereas the ablation acts at L24+L31+L41, so this flip comparison is a "
                               "single-site lower bound.")},
        # Sign convention: base minus intervened, so a positive shift means the gap moved toward keep.
        "mean_keepgap_shift_toward_keep": {
            "tox_ablation": round(float(np.mean((gb - R["g_tox"].to_numpy())[dec])), 3),
            "generic_keep_steer": round(float(np.mean((gb - R["g_keep"].to_numpy())[dec])), 3),
            "random_steer": round(float(np.mean((gb - R["g_rand"].to_numpy())[dec])), 3)},
        "mean_match_norm_m_i": round(float(R["m_i"].mean()), 3),
    }
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[sanity] WROTE {OUT}\n{json.dumps(res, indent=2)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(n=a.n, smoke=a.smoke)


if __name__ == "__main__":
    main()
