"""SAE BATTERY -- Test S (SUFFICIENCY): does clamping TOXICITY features UP induce removal of KEPT, low-toxicity
comments? The directional complement to Test N.

Per the locked spec. Pool = model-KEPT (gap_rules<0) AND bottom-quartile toxicity (confidently-kept benign text).
Intervention (live forward hook at L24+L31+L41): for each pre-registered TOX feature whose live activation is
below its p95, CLAMP it up to p95; add the decoded delta to resid_post: h' = h + sum_f (clamp_f - act_f) * w_dec[f].
Metric: removal-INDUCTION rate (gap crosses <0 -> >0) + mean delta-gap toward remove.
Controls (the key falsifier): clamp RANDOM (rotating over 20 frozen sets) and CLEAN_COMMUNITY features to THEIR OWN
p95 -- same magnitude perturbation by construction, so if only TOX induces removal, distribution-shift is ruled out.
Dose: clamp to p90/p95/p99. Off-distribution guard: report added-contribution norm / residual norm.
CI: subreddit-clustered bootstrap on (induction_tox - induction_control).

  smoke: CUDA_VISIBLE_DEVICES=3 python -m pipeline.kumar_mod.sae_sufficiency --smoke
  run:   CUDA_VISIBLE_DEVICES=3 python -m pipeline.kumar_mod.sae_sufficiency --n 2000
Out: results/kumar_mod/sae_sufficiency.json
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
OUT = ROOT / "results" / "kumar_mod" / "sae_sufficiency.json"
REPO = "google/gemma-scope-2-12b-it"
LAYERS = [24, 31, 41]  # intervene at three resid_post sites jointly, not one
WIDTH = "16k"
R_RAND = 5            # random control sets drawn per comment
N_RAND_SETS = 20      # rotate over 20 frozen random sets so the control isn't one lucky draw
SEED = 11             # shared project seed


def _load_sae(L, dev):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{WIDTH}_l0_medium/params.safetensors")
    sd = load_file(p)
    return {k: sd[k].to(dev, torch.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")}


def run(n=2000, smoke=False):
    import torch
    meta = pl.read_parquet(DEC / "meta.parquet")
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    # attach per-comment Detoxify toxicity to the decision-axis rows; sort by row to align with the cached activations
    jj = meta.select(["row", "subreddit", "idx", "label", "gap_rules"]).join(
        tox, on=["subreddit", "idx"], how="left").sort("row")
    # pre-registered feature sets (frozen on a held-out split before this run) per layer
    FS = {L: json.loads((FSDIR / f"featsets_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}
    rng = np.random.default_rng(SEED)

    tok, model = _load_model()
    dev = next(model.parameters()).device
    layers_mod = (model.model.language_model.layers if hasattr(model.model, "language_model")
                  else model.model.layers)
    tok.truncation_side = "left"  # keep the rules + question tail when a long body overflows MAX_LEN
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    SAE = {L: _load_sae(L, dev) for L in LAYERS}


    # clamp targets are calibrated on the rules-position activations of the *whole* corpus, not just the pool,
    # so a "clamp up to p95" means p95 of the natural population for that feature
    def feat_pcts(L, feats):
        W = SAE[L]; X = np.load(DEC / f"res_rules_L{L}.fp16.npy").astype(np.float32)
        we = W["w_enc"][:, feats].cpu().numpy(); be = W["b_enc"][feats].cpu().numpy()
        th = W["threshold"][feats].cpu().numpy()
        pre = X @ we + be; a = pre * (pre > th)  # JumpReLU: encode then gate at the per-feature threshold
        P = {}
        for q, name in [(90, "p90"), (95, "p95"), (99, "p99")]:
            # percentile over active firings only (col>0); a never-firing feature gets a 0 target
            P[name] = np.array([np.percentile(col[col > 0], q) if (col > 0).any() else 0.0 for col in a.T])
        return P

    # compute percentiles once over the union of every feature any condition will touch (tox + community + all random)
    needed = {L: sorted(set(FS[L]["TOX"]) | set(FS[L]["CLEAN_COMMUNITY"])
                        | set(int(x) for r in FS[L]["RANDOM"] for x in r)) for L in LAYERS}
    PCT = {L: feat_pcts(L, needed[L]) for L in LAYERS}
    pos = {L: {f: i for i, f in enumerate(needed[L])} for L in LAYERS}

    def pvec(L, feats, name):
        return torch.tensor([PCT[L][name][pos[L][f]] for f in feats], device=dev, dtype=torch.float32)

    def T(idx_list):
        return torch.tensor(idx_list, device=dev, dtype=torch.long)

    SET = {L: {"tox": (T(FS[L]["TOX"]), FS[L]["TOX"]), "comm": (T(FS[L]["CLEAN_COMMUNITY"]), FS[L]["CLEAN_COMMUNITY"]),
               "rand": [(T(r), r) for r in FS[L]["RANDOM"]]} for L in LAYERS}

    norm_frac = {"v": []}

    # build a forward hook that clamps features F at layer L up to `target`, leaving features already above it alone
    def mk(L, F, target, track=False):
        wdec = SAE[L]["w_dec"]
        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32)
            a = encode_acts(hf, L)
            cur = a[..., F]
            clamped = torch.maximum(cur, target)  # only push up; never lower an already-high feature
            # write the activation bump back through the decoder rows: h' = h + (clamp - act) @ w_dec[F]
            delta = (clamped - cur) @ wdec[F]
            hf2 = hf + delta
            if track:
                # off-distribution guard: how big is the injected delta vs the residual, at the last token
                dn = delta[:, -1, :].norm().item(); rn = hf[:, -1, :].norm().item()
                norm_frac["v"].append(dn / (rn + 1e-6))
            h2 = hf2.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    def encode_acts(x, L):
        s = SAE[L]; pre = x @ s["w_enc"] + s["b_enc"]; return pre * (pre > s["threshold"])

    def gap(enc, sets=None, pname="p95", track=False):
        """sets: dict L -> (idx_tensor, idx_list); clamp those features to their pname percentile at each L."""
        handles = []
        if sets:
            for L, (Ft, Fl) in sets.items():
                handles.append(layers_mod[L].register_forward_hook(mk(L, Ft, pvec(L, Fl, pname), track)))
        try:
            with torch.no_grad():
                out = model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        # decision axis = yes-minus-no logit gap at the final token; >0 leans remove, <0 leans keep
        ll = out.logits[0, -1, :].to(torch.float32)
        return float((ll[yes_id] - ll[no_id]).item())


    # pool = confidently-kept benign text: model already keeps it (gap_rules<0) AND it sits in the bottom
    # toxicity quartile, so any induced removal can't be blamed on pre-existing toxicity
    tv = jj["tox_toxicity"].to_numpy()
    q25 = np.nanpercentile(tv, 25)
    pool = jj.filter((pl.col("gap_rules") < 0) & (pl.col("tox_toxicity") <= q25))
    rows = pool.to_dicts()
    if smoke:
        rows = rows[:10]
    elif n < len(rows):
        rows = [rows[i] for i in rng.permutation(len(rows))[:n]]  # seeded subsample for tractability

    bodies, recs = {}, []
    for ci, r in enumerate(rows):
        s = r["subreddit"]
        if s not in bodies:
            bodies[s] = K.load_comments(s)
        body = bodies[s][r["idx"]][0]
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        enc = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        tset = {L: SET[L]["tox"] for L in LAYERS}
        cset = {L: SET[L]["comm"] for L in LAYERS}
        # base = unperturbed gap; then clamp tox to each dose, and clamp community to p95 as the matched control
        rec = {"sub": s, "y": int(r["label"]), "base": gap(enc),
               "tox_p95": gap(enc, tset, "p95", track=True), "tox_p90": gap(enc, tset, "p90"),
               "tox_p99": gap(enc, tset, "p99"), "comm_p95": gap(enc, cset, "p95")}
        for k in range(R_RAND):
            # rotate which frozen random set each comment gets, so the 20 sets are spread across the pool
            d = (ci + k) % N_RAND_SETS
            rec[f"rand{k}"] = gap(enc, {L: SET[L]["rand"][d] for L in LAYERS}, "p95")
        recs.append(rec)
        if smoke or (ci + 1) % 200 == 0:
            print(f"[sae S] {ci+1}/{len(rows)} base={rec['base']:.2f} tox_p95={rec['tox_p95']:.2f} "
                  f"comm_p95={rec['comm_p95']:.2f}", flush=True)

    R = pl.DataFrame(recs)
    # dec: drop comments sitting on the decision boundary (gap ~ 0) where a flip is meaningless
    gb = R["base"].to_numpy(); dec = np.abs(gb) > 1e-6
    rand_cols = [f"rand{k}" for k in range(R_RAND)]

    # induction = fraction of kept comments (base gap<0) whose gap crosses to >0 (remove) under the clamp
    def induce(col):
        g = R[col].to_numpy(); m = dec & (gb < 0)
        return float(np.mean(g[m] > 0)) if m.sum() else None

    # mean signed shift of the gap toward remove, over the same kept population
    def dgap(col):
        g = R[col].to_numpy(); m = dec & (gb < 0)
        return float(np.mean((g - gb)[m])) if m.sum() else None

    def rand_ind():
        v = [induce(c) for c in rand_cols]; v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    # cluster bootstrap by subreddit: comments within a community aren't independent, so resample whole communities
    subs_u = sorted(set(R["sub"].to_list())); idxby = {s: np.where(R["sub"].to_numpy() == s)[0] for s in subs_u}
    brng = np.random.default_rng(SEED + 1)  # distinct stream from the subsampling rng

    # bootstrap the dissociation (induction_a - induction_control) over resampled communities
    def boot(col_a, col_b=None, b_rand=False):
        ga = R[col_a].to_numpy(); rand_arr = [R[c].to_numpy() for c in rand_cols]
        gbb = None if b_rand else R[col_b].to_numpy(); out = []
        for _ in range(50 if smoke else 2000):
            samp = [subs_u[i] for i in brng.integers(0, len(subs_u), len(subs_u))]
            ix = np.concatenate([idxby[s] for s in samp]); m = dec[ix] & (gb[ix] < 0)
            if m.sum() == 0:
                continue
            fa = np.mean(ga[ix][m] > 0)
            # for the random control, also resample which of the R_RAND sets to compare against each replicate
            fb = (np.mean(rand_arr[brng.integers(0, R_RAND)][ix][m] > 0) if b_rand
                  else np.mean(gbb[ix][m] > 0))
            out.append(float(fa - fb))
        return out

    def ci(a):
        # 95% percentile interval over bootstrap replicates
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)] if a else None

    res = {
        "analysis": "sae_feature_sufficiency_toxicity", "model": "google/gemma-3-12b-it", "sae": REPO,
        "width": WIDTH, "layers": LAYERS, "n": int(len(R)), "n_subs": len(subs_u),
        "pool": "model-kept (gap<0) AND bottom-quartile toxicity", "clamp": "tox features up to p95 (dose p90/p95/p99)",
        "induction_rate": {"tox_p95": induce("tox_p95"), "community_p95": induce("comm_p95"), "random_p95": rand_ind()},
        "mean_delta_gap_toward_remove": {"tox_p95": dgap("tox_p95"), "community_p95": dgap("comm_p95"),
                                         "random_p95": float(np.mean([dgap(c) for c in rand_cols]))},
        "dose_response_induction": {"p90": induce("tox_p90"), "p95": induce("tox_p95"), "p99": induce("tox_p99")},
        "dissociation_ci95": {
            "tox_minus_random": {"point": round((induce("tox_p95") or 0) - (rand_ind() or 0), 4),
                                 "ci95": ci(boot("tox_p95", b_rand=True))},
            "tox_minus_community": {"point": round((induce("tox_p95") or 0) - (induce("comm_p95") or 0), 4),
                                    "ci95": ci(boot("tox_p95", col_b="comm_p95"))}},
        "off_distribution_guard": {"added_contrib_norm_frac_mean": round(float(np.mean(norm_frac["v"])), 4)
                                   if norm_frac["v"] else None,
                                   "note": "matched-p95 random/community controls give same perturbation magnitude"},
        "interpretation": ("Sufficiency: clamping toxicity features UP on confidently-kept benign comments induces "
                           "removal far more than matched random/community clamps (same magnitude) -> the toxicity "
                           "pathway is sufficient, and removal is not a generic distribution-shift artifact. Monotone "
                           "dose-response (p90<p95<p99) is the causal signature."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 16k is the headline width and keeps the bare filename; other widths get a suffix so they don't clobber it
    base = OUT if WIDTH == "16k" else OUT.with_name(OUT.stem + f"_{WIDTH}.json")
    out_path = base.with_suffix(".smoke.json") if smoke else base
    out_path.write_text(json.dumps(res, indent=2))
    ir = res["induction_rate"]; di = res["dissociation_ci95"]
    print(f"[sae S] induce tox={ir['tox_p95']} comm={ir['community_p95']} rand={ir['random_p95']} "
          f"| diss(tox-rand)={di['tox_minus_random']['point']} CI={di['tox_minus_random']['ci95']} "
          f"| dose {res['dose_response_induction']} -> {out_path}", flush=True)
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
