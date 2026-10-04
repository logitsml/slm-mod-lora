"""SAE NECESSITY -- position disentangling (gemma-3-12b-it). Answers: WHERE does the ~100% toxicity-ablation
flip come from? Ablate the SAME frozen toxicity features (L24+L31+L41) but restricted to one prompt segment at
a time, reusing the EXACT live hook of sae_causal_dosedense.

Prompt segments (Kumar 9-message chat):
  rules    : everything before the few-shot demo (the subreddit rules block)
  demo     : the toxic few-shot prime "You are a stupid idiot."   <-- the candidate confound
  comment  : the TARGET comment body
  decision : everything after the target comment (ACK + JSON request + generation scaffold = the readout site)
  comment_plus_decision : comment ∪ decision (the target's content AND where it is read out)
  all      : every position (= the headline 100% necessity flip)

Decisive reads:
  flip(demo) ≈ 0                          -> the demo does NOT drive it (confound refuted)
  flip(comment_plus_decision) ≈ flip(all) -> the target comment + its readout drives it (necessity holds, readout-localized)
  flip(comment) ≈ 0 but flip(decision) high -> toxicity is read out downstream of the short comment (encoded-vs-used)

  smoke: CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.sae_position_probe --smoke
  run:   CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.sae_position_probe --n 2000
Out: results/kumar_mod/sae_position_probe.json
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
OUT = ROOT / "results" / "kumar_mod" / "sae_position_probe.json"
REPO = "google/gemma-scope-2-12b-it"
LAYERS = [24, 31, 41]
WIDTH = "16k"
SEED = 11
# Literal substring of the toxic few-shot prime; located by exact string match to carve out the demo segment.
DEMO_STR = "You are a stupid idiot"
SEGS = ["rules", "demo", "comment", "decision", "comment_plus_decision"]


def _load_sae(L, dev):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{WIDTH}_l0_medium/params.safetensors")
    sd = load_file(p)
    # Keep SAE params in fp32 so encode/decode stays numerically stable regardless of the model's dtype.
    return {k: sd[k].to(dev, torch.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")}


def run(n=2000, smoke=False):
    import torch
    meta = pl.read_parquet(DEC / "meta.parquet")
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    # Align decision-axis rows with their Detoxify score by (subreddit, idx); sort by row keeps the original order.
    jj = meta.select(["row", "subreddit", "idx", "label", "gap_rules"]).join(
        tox, on=["subreddit", "idx"], how="left").sort("row")
    # Frozen toxicity feature ids per layer (selected upstream); reused verbatim so this probe ablates the SAME features.
    FS = {L: json.loads((FSDIR / f"featsets_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}
    rng = np.random.default_rng(SEED)

    tok, model = _load_model()
    dev = next(model.parameters()).device
    layers_mod = (model.model.language_model.layers if hasattr(model.model, "language_model")
                  else model.model.layers)
    tok.truncation_side = "left"
    # Readout is the yes-vs-no logit gap on the final token; cache both ids once.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    SAE = {L: _load_sae(L, dev) for L in LAYERS}

    def encode(x, L):
        # JumpReLU SAE: pre-activation gated by the learned per-feature threshold (values below threshold zeroed).
        s = SAE[L]; pre = x @ s["w_enc"] + s["b_enc"]; return pre * (pre > s["threshold"])

    def mk(L, F, pos):
        wdec = SAE[L]["w_dec"]
        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32)
            a = encode(hf, L)
            # Reconstruct only the toxicity features' contribution to the residual stream...
            contrib = a[..., F] @ wdec[F]
            if pos is not None:
                # ...and zero it everywhere except the targeted positions, so ablation is confined to one segment.
                pm = torch.zeros(hf.shape[1], device=hf.device, dtype=hf.dtype); pm[pos] = 1.0
                contrib = contrib * pm[None, :, None]
            # Subtract the toxicity component from the residual (project it out), leaving everything else intact.
            hf = hf - contrib
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    TOXSET = {L: torch.tensor(FS[L]["TOX"], device=dev, dtype=torch.long) for L in LAYERS}

    def gap(enc, pos="ALL"):
        handles = []
        if pos != "NONE":
            # pos="ALL" -> ablate every position (pos=None inside the hook); a tensor -> restrict to those positions.
            p = None if pos == "ALL" else pos
            for L in LAYERS:
                handles.append(layers_mod[L].register_forward_hook(mk(L, TOXSET[L], p)))
        try:
            with torch.no_grad():
                out = model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        ll = out.logits[0, -1, :].to(torch.float32)
        return float((ll[yes_id] - ll[no_id]).item())

    def base_and_act(enc, segs):
        """Clean forward: base gap + per-segment toxicity-feature ACTIVATION MASS (summed over L24/31/41).
        This is the inertness diagnostic: if a segment's ablation is inert because little tox activation
        fires there, inertness is the expected outcome rather than a failure of the position mask."""
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        ll = out.logits[0, -1, :].to(torch.float32)
        g = float((ll[yes_id] - ll[no_id]).item())
        hs = out.hidden_states
        # Per-token toxicity-feature activation, summed across the three layers. hidden_states[L+1] is resid_post of
        # layer L (index 0 is the embedding), matching the resid_post SAEs.
        per_pos = None
        for L in LAYERS:
            res = hs[L + 1][0].to(torch.float32)
            a = encode(res, L)[:, TOXSET[L]].sum(1)
            per_pos = a if per_pos is None else per_pos + a
        per_pos = per_pos.detach().cpu().numpy()
        total = float(per_pos.sum()) + 1e-9
        act = {k: float(per_pos[segs[k]].sum()) for k in SEGS}
        share = {k: round(act[k] / total, 4) for k in SEGS}
        return g, act, share

    def seg_positions(p, body, ofs):
        # Map character spans in the prompt to token indices via the tokenizer offset mapping.
        out = {k: [] for k in SEGS}
        d0 = p.find(DEMO_STR)
        # rfind: the comment body can recur (e.g. echoed in rules); the last occurrence is the actual target slot.
        c0 = p.rfind(body) if body else -1
        c1 = c0 + len(body) if c0 >= 0 else -1
        # A token belongs to [lo, hi) if its char span overlaps it; b > a drops zero-width special tokens.
        def toks(lo, hi):
            return [i for i, (a, b) in enumerate(ofs) if (b > lo and a < hi and b > a)]
        if d0 >= 0:
            out["demo"] = toks(d0, d0 + len(DEMO_STR))
            # Rules = everything strictly before the demo prime.
            out["rules"] = [i for i, (a, b) in enumerate(ofs) if (b <= d0 and b > a)]
        if c0 >= 0:
            out["comment"] = toks(c0, c1)
            # Decision = everything at or after the comment's end (ACK + JSON request + generation scaffold).
            out["decision"] = [i for i, (a, b) in enumerate(ofs) if (a >= c1 and b > a)]
            out["comment_plus_decision"] = sorted(set(out["comment"]) | set(out["decision"]))
        return out

    # Restrict to cases the rules block already removed (gap_rules>0): these are where toxicity-ablation can flip the
    # keep/remove decision, so they isolate the necessity effect the probe localizes.
    pool = jj.filter(pl.col("gap_rules") > 0)
    rows = pool.to_dicts()
    if smoke:
        rows = rows[:10]
    elif n < len(rows):
        # Seed-11 permutation subsample for a reproducible n-row draw.
        rows = [rows[i] for i in rng.permutation(len(rows))[:n]]

    bodies, recs, nseg = {}, [], {k: 0 for k in SEGS}
    for ci, r in enumerate(rows):
        s = r["subreddit"]
        if s not in bodies:
            bodies[s] = K.load_comments(s)
        body = bodies[s][r["idx"]][0]
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        ofs = tok(p, truncation=True, max_length=MAX_LEN, add_special_tokens=False,
                  return_offsets_mapping=True)["offset_mapping"]
        enc = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        seqlen = enc["input_ids"].shape[1]
        # Offsets and input_ids must align 1:1; on any length mismatch (e.g. truncation edge case) skip segment
        # resolution for this row rather than risk a misaligned position mask.
        segs = seg_positions(p, body, ofs) if len(ofs) == seqlen else {k: [] for k in SEGS}
        bg, actmass, _ = base_and_act(enc, segs)
        rec = {"sub": s, "y": int(r["label"]), "base": bg, "all": gap(enc, "ALL")}
        for k in SEGS:
            rec["act_" + k] = actmass[k]
            if segs[k]:
                nseg[k] += 1
                rec[k] = gap(enc, torch.tensor(segs[k], device=dev, dtype=torch.long))
            else:
                # Empty segment (not found / unresolved) -> no ablation, so its gap equals the clean base gap.
                rec[k] = bg
        rec["_seglens"] = {k: len(segs[k]) for k in SEGS}
        recs.append(rec)
        if smoke or (ci + 1) % 200 == 0:
            sl = rec["_seglens"]
            print(f"[pos] {ci+1}/{len(rows)} base={rec['base']:.1f} all={rec['all']:.1f} "
                  f"demo={rec['demo']:.1f} rules={rec['rules']:.1f} comment={rec['comment']:.1f} "
                  f"decision={rec['decision']:.1f} c+d={rec['comment_plus_decision']:.1f} "
                  f"seglens={sl}", flush=True)

    R = pl.DataFrame([{k: v for k, v in r.items() if k != "_seglens"} for r in recs])
    # Only rows with a non-degenerate base gap have a defined decision sign to flip.
    gb = R["base"].to_numpy(); dec = np.abs(gb) > 1e-6
    subs_u = sorted(set(R["sub"].to_list())); idxby = {s: np.where(R["sub"].to_numpy() == s)[0] for s in subs_u}
    # Separate stream (SEED+1) for the bootstrap so resampling doesn't share state with the row subsample draw.
    brng = np.random.default_rng(SEED + 1)

    def flip(col, mask=None):
        # Flip rate = fraction of decided rows whose yes/no sign changes under this segment's ablation vs the base.
        g = R[col].to_numpy(); m = dec if mask is None else (dec & mask)
        return float(np.mean(np.sign(g[m]) != np.sign(gb[m]))) if m.sum() else None

    def boot_ci(col):
        # Subreddit-clustered bootstrap: resample whole communities with replacement (not individual rows) so the CI
        # respects within-community correlation. 2000 reps; 95% CI from the 2.5/97.5 percentiles.
        g = R[col].to_numpy(); out = []
        for _ in range(50 if smoke else 2000):
            samp = [subs_u[i] for i in brng.integers(0, len(subs_u), len(subs_u))]
            ix = np.concatenate([idxby[s] for s in samp]); d = dec[ix]
            if d.sum() == 0:
                continue
            out.append(float(np.mean(np.sign(g[ix][d]) != np.sign(gb[ix][d]))))
        return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)] if out else None

    arms = ["all"] + SEGS
    res = {"analysis": "sae_necessity_position_disentangle", "model": "google/gemma-3-12b-it", "sae": REPO,
           "width": WIDTH, "layers": LAYERS, "n": int(len(R)), "n_subs": len(subs_u),
           "pool": "model-removed (gap_rules>0)",
           "n_segment_resolved": nseg,
           "flip_rate": {a: flip(a) for a in arms},
           "flip_ci95": {a: boot_ci(a) for a in arms},
           "mean_seglen": {k: round(float(np.mean([r["_seglens"][k] for r in recs])), 1) for k in SEGS},
           "interpretation": ("flip(demo)~0 => few-shot demo not the driver (confound refuted); "
                              "flip(comment_plus_decision)~flip(all) => target comment + its readout drives it; "
                              "flip(comment)<<flip(decision) => toxicity read out downstream of the comment.")}


    seg_act = {k: round(float(np.mean([r["act_" + k] for r in recs])), 3) for k in SEGS}
    tot = sum(seg_act.values()) + 1e-9
    res["tox_activation_mean_by_segment"] = seg_act
    res["tox_activation_share_by_segment"] = {k: round(seg_act[k] / tot, 4) for k in SEGS}
    # Firing-restricted control: restrict comment-ablation to comments whose own toxicity-feature firing is in the
    # top quartile; if those flip while the rest do not, comment-inertness simply reflects low firing.
    ca = R["act_comment"].to_numpy()
    hi = (ca >= np.percentile(ca[dec], 75)) if dec.sum() else np.zeros(len(ca), bool)
    res["comment_ablation_firing_check"] = {
        "flip_comment_all": flip("comment"),
        "flip_comment_top_quartile_body_toxic": flip("comment", hi),
        "mean_comment_tox_activation": round(float(np.mean(ca[dec])), 3) if dec.sum() else None,
        "note": ("Manipulation check. If comment-ablation flips the body-toxic top quartile but not the rest, "
                 "comment-inertness simply reflects low toxicity-feature firing on most comments, as expected. "
                 "A flat top quartile would instead indicate the intervention is not reaching the comment span.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUT.with_suffix(".smoke.json") if smoke else OUT
    out_path.write_text(json.dumps(res, indent=2))
    fr = res["flip_rate"]
    print(f"[pos] FLIPS all={fr['all']} demo={fr['demo']} rules={fr['rules']} comment={fr['comment']} "
          f"decision={fr['decision']} c+d={fr['comment_plus_decision']} -> {out_path}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    run(n=a.n, smoke=a.smoke)


if __name__ == "__main__":
    main()
