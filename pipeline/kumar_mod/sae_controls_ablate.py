"""SAE EXTRA-CONTROLS ablation (gemma-3-12b-it necessity test).

Reuses the EXACT live-ablation machinery of sae_causal_dosedense (same _load_model, _build_prompt, SAE load,
hook that subtracts a*(acts[F]@w_dec[F]) at L24+L31+L41, same flip definition, same subreddit-clustered
bootstrap). Adds the five control arms (frozen by sae_controls_freeze):
  TOX vs ACTFREQ_MATCHED / DECNORM_MATCHED / DLA_MATCHED / PERMLABEL  -- ablate-on-all-decided dissociations
  HELDOUT_TOX                                                          -- features chosen on half A, flip measured on the disjoint half B
Decoder vectors are unit-norm in Gemma Scope, so DECNORM_MATCHED is reported but expected to behave like any
matched non-tox set (the operative magnitude control is ACTFREQ/p95).

  smoke: CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.sae_controls_ablate --smoke
  run:   CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.sae_controls_ablate --n 2000
Out: results/kumar_mod/sae_controls_extra.json
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
OUT = ROOT / "results" / "kumar_mod" / "sae_controls_extra.json"
REPO = "google/gemma-scope-2-12b-it"
LAYERS = [24, 31, 41]  # ablate at all three jointly; necessity is a multi-layer effect
WIDTH = "16k"
SEED = 11  # shared project seed; pins both the row subsample and the bootstrap resampler
ARMS = ["tox", "actfreq", "decnorm", "dla", "perm", "held"]
CTRL_KEY = {"actfreq": "ACTFREQ_MATCHED", "decnorm": "DECNORM_MATCHED", "dla": "DLA_MATCHED",
            "perm": "PERMLABEL", "held": "HELDOUT_TOX"}


def _load_sae(L, dev):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{WIDTH}_l0_medium/params.safetensors")
    sd = load_file(p)
    # keep SAE params in fp32 regardless of model dtype: the encode/decode arithmetic and the
    # JumpReLU threshold comparison are sensitive to bf16 rounding
    return {k: sd[k].to(dev, torch.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")}


def run(n=2000, smoke=False):
    import torch
    meta = pl.read_parquet(DEC / "meta.parquet")
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    jj = meta.select(["row", "subreddit", "idx", "label", "gap_rules"]).join(
        tox, on=["subreddit", "idx"], how="left").sort("row")
    FS = {L: json.loads((FSDIR / f"featsets_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}
    EX = {L: json.loads((FSDIR / f"controls_extra_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}
    held_split = json.loads((FSDIR / "controls_heldout_split.json").read_text())
    held_B = {(r["subreddit"], int(r["idx"])) for r in held_split["heldout_B"]}

    # guard against drift in the frozen artifacts: the control file's re-selected TOX set must be
    # identical to the locked TOX set this ablation uses, or the dissociation isn't measuring the
    # same features
    for L in LAYERS:
        assert set(EX[L]["TOX_reselected_here"]) == set(FS[L]["TOX"]), f"L{L} control TOX != locked TOX"
        for c in ("ACTFREQ_MATCHED", "DECNORM_MATCHED", "DLA_MATCHED", "PERMLABEL", "HELDOUT_TOX"):
            assert len(EX[L][c]) >= 1, f"L{L} {c} empty"
    rng = np.random.default_rng(SEED)

    tok, model = _load_model()
    dev = next(model.parameters()).device
    layers_mod = (model.model.language_model.layers if hasattr(model.model, "language_model")
                  else model.model.layers)
    tok.truncation_side = "left"  # keep the trailing decision region; drop overflow from the prompt head
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    SAE = {L: _load_sae(L, dev) for L in LAYERS}

    def encode(x, L):
        # JumpReLU: gate the pre-activation by the learned per-feature threshold (not a soft ReLU)
        s = SAE[L]; pre = x @ s["w_enc"] + s["b_enc"]; return pre * (pre > s["threshold"])

    def T(idx_list):
        return torch.tensor(idx_list, device=dev, dtype=torch.long)

    SET = {L: {"tox": T(FS[L]["TOX"]),
               "actfreq": T(EX[L]["ACTFREQ_MATCHED"]), "decnorm": T(EX[L]["DECNORM_MATCHED"]),
               "dla": T(EX[L]["DLA_MATCHED"]), "perm": T(EX[L]["PERMLABEL"]),
               "held": T(EX[L]["HELDOUT_TOX"])} for L in LAYERS}

    def mk(L, F, pos=None, alpha=1.0):
        # forward hook that subtracts the feature-set F's SAE reconstruction from resid_post at layer L,
        # i.e. project the residual stream off those decoder directions in-place during the forward pass
        wdec = SAE[L]["w_dec"]
        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32)
            a = encode(hf, L)
            contrib = a[..., F] @ wdec[F]  # sum_{f in F} act_f * w_dec_f : the part of the edit owned by F
            if pos is not None:
                # restrict the ablation to specific sequence positions (target-comment vs rest)
                pm = torch.zeros(hf.shape[1], device=hf.device, dtype=hf.dtype)
                pm[pos] = 1.0
                contrib = contrib * pm[None, :, None]
            hf = hf - alpha * contrib
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    def gap(enc, layer_sets=None, pos=None):
        handles = []
        if layer_sets:
            for L, F in layer_sets.items():
                handles.append(layers_mod[L].register_forward_hook(mk(L, F, pos)))
        try:
            with torch.no_grad():
                out = model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        ll = out.logits[0, -1, :].to(torch.float32)
        return float((ll[yes_id] - ll[no_id]).item())  # decision axis = logit(yes) - logit(no)

    # restrict to comments the unablated model already decided to remove (positive rule-gap), so a
    # "flip" can only mean remove -> keep -- the direction toxicity ablation is hypothesised to cause
    pool = jj.filter(pl.col("gap_rules") > 0)
    rows = pool.to_dicts()
    if smoke:
        rows = rows[:10]
    elif n < len(rows):
        rows = [rows[i] for i in rng.permutation(len(rows))[:n]]

    def all3_tox():
        return {L: SET[L]["tox"] for L in LAYERS}
    bodies, recs, n_tgt_ok = {}, [], 0
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
        # locate the target comment's own tokens via char offsets: rfind because the body can also
        # appear earlier (e.g. echoed in a rule), and we want the last/decision-side occurrence.
        # len(ofs)==seqlen guards against truncation having dropped the matched span.
        bstart = p.rfind(body) if body else -1
        tlist = []
        if bstart >= 0 and len(ofs) == seqlen:
            bend = bstart + len(body)
            # token overlaps [bstart,bend) and is non-empty (b>a drops special/zero-width offsets)
            tlist = [i for i, (a, b) in enumerate(ofs) if (b > bstart and a < bend and b > a)]
        tgt_ok = len(tlist) > 0
        n_tgt_ok += int(tgt_ok)
        tpos = torch.tensor(tlist, device=dev, dtype=torch.long) if tgt_ok else None
        ntpos = (torch.tensor([i for i in range(seqlen) if i not in set(tlist)], device=dev, dtype=torch.long)
                 if tgt_ok else None)
        # base = unablated gap; then one ablated gap per arm (tox + the five controls), all at L24/31/41
        rec = {"sub": s, "y": int(r["label"]), "in_B": (s, int(r["idx"])) in held_B, "tgt_ok": tgt_ok,
               "base": gap(enc)}
        for arm in ARMS:
            rec[arm] = gap(enc, {L: SET[L][arm] for L in LAYERS})
        if tgt_ok:
            # position-localized contrast: ablate tox only inside the target comment vs only outside it
            rec["tox_target"] = gap(enc, all3_tox(), pos=tpos)
            rec["tox_nontarget"] = gap(enc, all3_tox(), pos=ntpos)
        else:
            # no resolvable span: fall back to the all-position tox gap so these rows drop out under
            # the tgt mask rather than biasing the localization contrast
            rec["tox_target"] = rec["tox"]; rec["tox_nontarget"] = rec["tox"]
        recs.append(rec)
        if smoke or (ci + 1) % 200 == 0:
            print(f"[ctrlN] {ci+1}/{len(rows)} base={rec['base']:.2f} tox={rec['tox']:.2f} "
                  f"tox_target={rec['tox_target']:.2f} tox_nontarget={rec['tox_nontarget']:.2f} "
                  f"perm={rec['perm']:.2f} held={rec['held']:.2f} (tgt_span={len(tlist)}/{seqlen})", flush=True)
    print(f"[ctrlN] target span resolved for {n_tgt_ok}/{len(rows)} comments", flush=True)

    R = pl.DataFrame(recs)
    gb = R["base"].to_numpy()
    dec = np.abs(gb) > 1e-6  # "decided": base gap is off the fence, so a sign change is meaningful
    y = R["y"].to_numpy()
    inB = R["in_B"].to_numpy(); tgt = R["tgt_ok"].to_numpy()

    def flip(col, mask=None):
        # flip = sign of the gap changed vs base (remove <-> keep); evaluated only on decided rows
        g = R[col].to_numpy(); m = dec if mask is None else (dec & mask)
        return float(np.mean(np.sign(g[m]) != np.sign(gb[m]))) if m.sum() else None

    def dgap(col, mask=None):
        g = R[col].to_numpy(); m = dec if mask is None else (dec & mask)
        return float(np.mean((gb - g)[m])) if m.sum() else None

    subs_u = sorted(set(R["sub"].to_list())); idxby = {s: np.where(R["sub"].to_numpy() == s)[0] for s in subs_u}
    brng = np.random.default_rng(SEED + 1)  # separate stream from the subsample rng for clean reproducibility

    def boot_diss(col_a, col_b, mask=None):
        # subreddit-clustered bootstrap on the flip-rate difference (arm A minus arm B): resample whole
        # communities with replacement so the CI respects within-community correlation, not per-comment
        ga = R[col_a].to_numpy(); gbb = R[col_b].to_numpy(); out = []
        base_mask = dec if mask is None else (dec & mask)
        for _ in range(50 if smoke else 2000):  # 2000 reps is the project-wide CI convention
            samp = [subs_u[i] for i in brng.integers(0, len(subs_u), len(subs_u))]
            ix = np.concatenate([idxby[s] for s in samp]); d = base_mask[ix]
            if d.sum() == 0:
                continue
            fa = np.mean(np.sign(ga[ix][d]) != np.sign(gb[ix][d]))
            fb = np.mean(np.sign(gbb[ix][d]) != np.sign(gb[ix][d]))
            out.append(float(fa - fb))
        return out

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)] if a else None

    # report flips overall and split by the human gold label, so a flip-to-keep can be read against
    # whether the comment was actually benign (y==0) or genuinely removable (y==1)
    strata = {"all": None, "human_keep": (y == 0), "human_remove": (y == 1)}
    matchq = {f"L{L}": EX[L]["match_quality"] for L in LAYERS}
    res = {
        "analysis": "sae_extra_controls_necessity", "model": "google/gemma-3-12b-it", "sae": REPO,
        "width": WIDTH, "layers": LAYERS, "n": int(len(R)), "n_subs": len(subs_u),
        "n_in_B": int((dec & inB).sum()), "pool": "model-removed (gap_rules>0)",
        "flip_rate": {st: {arm: flip(arm, m) for arm in ARMS} | {"base_decided": int((dec if m is None else dec & m).sum())}
                      for st, m in strata.items()},
        "mean_delta_gap_toward_keep": {arm: dgap(arm) for arm in ARMS},
        "dissociation_tox_minus_control_ci95": {
            ctrl: {"point": round((flip("tox") or 0) - (flip(ctrl) or 0), 4),
                   "ci95": ci(boot_diss("tox", ctrl))}
            for ctrl in ["actfreq", "decnorm", "dla", "perm"]},
        # anti-circularity test: HELDOUT_TOX features were chosen on half A, flip is scored only on the
        # disjoint half B (inB mask), so selection and evaluation never touch the same comments
        "heldout_on_disjoint_B": {
            "flip_tox_on_B": flip("tox", inB), "flip_heldout_on_B": flip("held", inB),
            "flip_perm_on_B": flip("perm", inB),
            "diss_heldout_minus_perm_on_B": {
                "point": round((flip("held", inB) or 0) - (flip("perm", inB) or 0), 4),
                "ci95": ci(boot_diss("held", "perm", mask=inB))},
            "diss_heldout_minus_tox_on_B": {
                "point": round((flip("held", inB) or 0) - (flip("tox", inB) or 0), 4),
                "ci95": ci(boot_diss("held", "tox", mask=inB))},
            "note": "HELDOUT_TOX features were selected on disjoint half A; flip measured on half B => not selection-on-eval circular. heldout-minus-tox near 0 = no circularity penalty; heldout-minus-perm > 0 = real toxicity selection beats the null."},
        "position_localization": {
            "n_target_resolved": int((dec & tgt).sum()),
            "flip_tox_all_positions": flip("tox", tgt), "flip_tox_target_only": flip("tox_target", tgt),
            "flip_tox_nontarget_only": flip("tox_nontarget", tgt),
            "diss_target_minus_nontarget": {
                "point": round((flip("tox_target", tgt) or 0) - (flip("tox_nontarget", tgt) or 0), 4),
                "ci95": ci(boot_diss("tox_target", "tox_nontarget", mask=tgt))},
            "note": "Ablating toxicity ONLY in the target comment vs ONLY outside it (rules, the few-shot demo, and the post-comment decision region). Observed: non-target-only matches all-positions (~1.0) while target-only is near zero, so by the ablated layers (L24+) the toxicity signal driving the flip has already propagated to the post-comment readout positions. The flip is not produced by damaging the target comment's own tokens, consistent with the position-resolved probe placing necessity in the decision region."},
        "match_quality_from_freeze": matchq,
        "interpretation": ("Each control isolates one alternative explanation for the ~100% toxicity flip: "
                           "actfreq=firing-rate, decnorm=residual-edit-size (VACUOUS: Gemma Scope decoders are unit-norm, "
                           "so this is just another matched non-tox set), dla=decision-readout magnitude (matched at L24/L31; "
                           "at L41 no non-tox feature reaches toxicity's DLA, so the L41 DLA control is a LOWER BOUND, not a "
                           "true match -- itself evidence that toxicity uniquely owns the readout), perm=selection-procedure null "
                           "(check rgap_perm_mean: a clean null also has low r_gap), held=selection circularity. "
                           "Toxicity necessity is robust iff flip(tox)-flip(control) CI95>0 for the matched controls, "
                           "perm flips ~like random, and heldout_tox still flips B (heldout-minus-tox CI ~ 0)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUT.with_suffix(".smoke.json") if smoke else OUT
    out_path.write_text(json.dumps(res, indent=2))
    fr = res["flip_rate"]["all"]; di = res["dissociation_tox_minus_control_ci95"]; hb = res["heldout_on_disjoint_B"]
    print(f"[ctrlN] flips tox={fr['tox']} actfreq={fr['actfreq']} decnorm={fr['decnorm']} dla={fr['dla']} "
          f"perm={fr['perm']} held={fr['held']}", flush=True)
    print(f"[ctrlN] diss tox-dla={di['dla']['point']} CI={di['dla']['ci95']} | tox-actfreq={di['actfreq']['point']} "
          f"CI={di['actfreq']['ci95']} | tox-perm={di['perm']['point']} CI={di['perm']['ci95']}", flush=True)
    print(f"[ctrlN] heldout-on-B: tox={hb['flip_tox_on_B']} held={hb['flip_heldout_on_B']} perm={hb['flip_perm_on_B']} "
          f"(n_B={res['n_in_B']})", flush=True)
    pl_ = res["position_localization"]
    print(f"[ctrlN] position: tox_all={pl_['flip_tox_all_positions']} target_only={pl_['flip_tox_target_only']} "
          f"nontarget_only={pl_['flip_tox_nontarget_only']} (n_tgt={pl_['n_target_resolved']}) "
          f"diss_tgt-nontgt={pl_['diss_target_minus_nontarget']['point']} CI={pl_['diss_target_minus_nontarget']['ci95']} "
          f"-> {out_path}", flush=True)
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
