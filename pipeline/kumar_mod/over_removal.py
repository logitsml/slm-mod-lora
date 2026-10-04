"""Over-removal / silencing analysis (the second failure direction).

Question: among comments the community KEPT (label==0), do prompted LLMs disproportionately
flag the toxic-scoring ones for removal, silencing norm-compliant-but-toxic speech? This
completes the 2x2 whose other half (under-removal of non-toxic norm violations) the encoder
section already covers.

Estimands, fixed before computation:

  PRIMARY (threshold-robust)  Toxicity-selectivity ratio per method
      S = P(flag | kept, toxic) / P(flag | kept, non-toxic)
    flag = the method's committed remove decision; kept = recorded label 0. Because both terms
    are within one method's operating point, S is invariant to how aggressive that method is
    overall, which is the confound 9.2 warns about (the encoder is more liberal at the shared
    0.5 cut). S >> 1 means the method over-removes kept speech specifically when it scores toxic.

  SECONDARY (strictness-matched)  False-flag rate on kept-toxic at EQUAL overall removal rate.
    Each method's continuous score is thresholded so its overall removal rate on the test fold
    equals a common target r; we then report P(score >= t_m | kept, toxic). This compares
    methods at equal aggressiveness directly.

Toxic definition: Detoxify >= 0.5, with a {0.3, 0.7} sweep, a top-decile-within-community
variant, and a ToxiGen >= 0.5 robustness replication. CIs are subreddit-clustered bootstrap
(B=2000, seed 11). All arms are compared on the shared seed-11 test fold, intersected to the
rows present for every arm.

Out: results/kumar_mod/analysis/over_removal.json
     results/kumar_mod/analysis/over_removal_communities.json   (empirical insult-tolerant set)
CPU only.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from pipeline.kumar_mod import fairness_compare as FC
from pipeline.kumar_mod._common import encoder_test_probs

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "kumar_mod"
PROC = ROOT / "data" / "processed"
OUT = RES / "analysis" / "over_removal.json"
OUT_COMM = RES / "analysis" / "over_removal_communities.json"

SEED = 11
B = 2000
THRESH_MAIN = 0.5
THRESH_SWEEP = (0.3, 0.5, 0.7)  # 0.3/0.7 bracket the 0.5 toxic cut to show the ratio is not an artifact of where we draw the line
# Each family: (binary remove-decision parquet, continuous logit-gap parquet). Gap is the
# strictness-matched secondary score; the binary file gives the committed remove flag.
LLM_FAMS = {"llm_gemma": ("llm_gemma.parquet", "llm_gap_gemma3_12b.parquet"),
            "llm_llama": ("llm_llama.parquet", "llm_gap_llama31_8b.parquet"),
            "llm_qwen": ("llm_qwen.parquet", "llm_gap_qwen25_7b.parquet")}


def load_methods():
    """Return {method: DataFrame(subreddit, idx, label, flag, score)} on the test fold,
    intersected to the rows present for every arm. flag = committed remove (0/1);
    score = continuous remove score (encoder prob, SLM/LLM logit gap)."""
    split = pl.read_parquet(FC.SPLIT)
    test = split.filter(pl.col("fold") == "test").select("subreddit", "idx", "label")
    methods = {}

    # Encoder commits at the shared 0.5 cut (same operating point used everywhere else);
    # the raw probability doubles as the continuous score for the strictness-matched arm.
    enc = encoder_test_probs().select(
        "subreddit", "idx",
        (pl.col("p") >= 0.5).cast(pl.Int64).alias("flag"), pl.col("p").alias("score"))
    methods["encoder_e5"] = test.join(enc, on=["subreddit", "idx"], how="inner")

    slm = pl.read_parquet(RES / "balanced" / "slm_mod_test.parquet").select(
        "subreddit", "idx",
        pl.col("would_moderate").cast(pl.Int64).alias("flag"), pl.col("gap").alias("score"))
    methods["slm_mod"] = test.join(slm, on=["subreddit", "idx"], how="inner")

    for name, (binf, gapf) in LLM_FAMS.items():
        b = pl.read_parquet(RES / "balanced" / binf).select(
            "subreddit", "idx", pl.col("would_moderate").cast(pl.Float64).alias("wm"))
        g = pl.read_parquet(RES / gapf).select("subreddit", "idx", pl.col("gap").alias("score"))
        # Drop rows whose remove decision failed to parse (non-finite wm) before casting to 0/1.
        d = (test.join(b, on=["subreddit", "idx"], how="inner")
                 .join(g, on=["subreddit", "idx"], how="inner")
                 .filter(pl.col("wm").is_finite())
                 .with_columns(pl.col("wm").cast(pl.Int64).alias("flag")).drop("wm"))
        methods[name] = d

    # Intersect every arm to a common row set so all methods are scored on identical comments;
    # an LLM family that dropped some unparseable rows would otherwise shrink the comparison silently.
    common = None
    for d in methods.values():
        keys = d.select("subreddit", "idx")
        common = keys if common is None else common.join(keys, on=["subreddit", "idx"], how="inner")
    methods = {m: d.join(common, on=["subreddit", "idx"], how="inner").sort("subreddit", "idx")
               for m, d in methods.items()}
    return methods, common.height


def tox_frame(which, common):
    if which == "ToxiGen":
        t = pl.read_parquet(PROC / "kumar_balanced_multitox.parquet").select(
            "subreddit", "idx", pl.col("tox_toxigen").alias("tox"))
    else:
        t = pl.read_parquet(PROC / "kumar_balanced_tox_sent.parquet").select(
            "subreddit", "idx", pl.col("tox_toxicity").alias("tox"))
    return common.join(t, on=["subreddit", "idx"], how="inner")


def selectivity_ratio(df, flag, kept_mask, toxic_mask):
    # Both rates conditioned on kept (label 0), so the ratio cancels the method's overall
    # aggressiveness; what survives is how much more it flags toxic-scoring kept speech.
    kt = kept_mask & toxic_mask
    knt = kept_mask & (~toxic_mask)
    if kt.sum() < 1 or knt.sum() < 1:
        return None
    p_tox = flag[kt].mean()
    p_non = flag[knt].mean()
    if p_non <= 0:  # zero denominator -> ratio undefined for this resample/def
        return None
    return float(p_tox / p_non), float(p_tox), float(p_non), int(kt.sum()), int(knt.sum())


def clustered_boot(subs_arr, fn):
    """fn(mask) -> scalar; resample communities with replacement."""
    # Resample whole subreddits, not rows: comments within a community share its norms and
    # are not independent, so a row bootstrap would understate the CI. Seed fixed for reproducibility.
    uniq = np.unique(subs_arr)
    byc = {s: np.where(subs_arr == s)[0] for s in uniq}
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(B):
        samp = uniq[rng.integers(0, len(uniq), len(uniq))]
        idx = np.concatenate([byc[s] for s in samp])  # all rows of each drawn community
        v = fn(idx)
        if v is not None and np.isfinite(v):  # skip resamples where the stat is undefined
            out.append(v)
    if not out:
        return None
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def main():
    methods, n_common = load_methods()
    # All arms share the same sorted (subreddit, idx) order, so base's label/subs arrays align
    # positionally with every method's flag/score array -- masks built here apply to all of them.
    base = methods["encoder_e5"].select("subreddit", "idx", "label")
    subs = base["subreddit"].to_numpy()
    label = base["label"].to_numpy()
    kept = label == 0  # the community kept it; the silencing question is only about these

    res = {"analysis": "over_removal_silencing", "n_common_rows": n_common,
           "n_methods": len(methods), "definition": {
               "ratio": "P(flag|kept,toxic) / P(flag|kept,non-toxic); flag=committed remove, kept=label0",
               "matched": "P(score>=t_m | kept, toxic) with t_m set to a common overall removal rate"},
           "by_tox_def": {}}

    flags = {m: methods[m]["flag"].to_numpy().astype(float) for m in methods}
    scores = {m: methods[m]["score"].to_numpy().astype(float) for m in methods}

    defs = {}
    for which in ("Detoxify", "ToxiGen"):
        # Re-sort the toxicity join the same way so tox[] lines up with the positional masks;
        # the height assert guards against any row silently dropped in the join.
        tf = tox_frame(which, base.select("subreddit", "idx")).sort("subreddit", "idx")
        assert tf.height == base.height, f"{which} toxicity join dropped rows"
        tox = tf["tox"].to_numpy()
        # ToxiGen is the identity-robust replication, run only at its 0.5 cut; Detoxify sweeps.
        for thr in (THRESH_SWEEP if which == "Detoxify" else (0.5,)):
            defs[f"{which}>={thr}"] = tox >= thr
        if which == "Detoxify":
            # Relative definition: "toxic" = top 10% of a community's own score distribution,
            # so the toxic set is non-empty even in communities whose absolute scores never reach 0.5.
            topdec = np.zeros(len(tox), dtype=bool)
            for s in np.unique(subs):
                ix = np.where(subs == s)[0]
                if len(ix) >= 10:  # need enough rows for a meaningful within-community decile
                    cut = np.quantile(tox[ix], 0.9)
                    topdec[ix] = tox[ix] >= cut
            defs["Detoxify_top_decile_per_comm"] = topdec

    for dname, toxic in defs.items():
        block = {"methods": {}}
        for m in methods:
            r = selectivity_ratio(None, flags[m], kept, toxic)
            if r is None:
                block["methods"][m] = {"ratio": None}
                continue
            ratio, p_tox, p_non, n_kt, n_knt = r
            def ratio_stat(ix, m=m):
                fm = flags[m][ix]; km = kept[ix]; tm = toxic[ix]
                kt, knt = km & tm, km & ~tm
                if kt.sum() < 1 or knt.sum() < 1 or fm[knt].mean() <= 0:
                    return None
                return fm[kt].mean() / fm[knt].mean()
            ci = clustered_boot(subs, ratio_stat)
            block["methods"][m] = {
                "selectivity_ratio": round(ratio, 3), "ratio_ci95": ci,
                "p_flag_kept_toxic": round(p_tox, 4), "p_flag_kept_nontoxic": round(p_non, 4),
                "n_kept_toxic": n_kt, "n_kept_nontoxic": n_knt}
        # Paired LLM-vs-encoder gap in the selectivity ratio, CI'd on the same community resamples
        # so the difference is tested directly rather than from two overlapping intervals.
        contrasts = {}
        for m in LLM_FAMS:
            def diff(ix, m=m):
                fm, fe = flags[m][ix], flags["encoder_e5"][ix]
                km, tm = kept[ix], toxic[ix]
                kt, knt = km & tm, km & ~tm
                if kt.sum() < 1 or knt.sum() < 1:
                    return None
                dm, de = fm[knt].mean(), fe[knt].mean()  # each arm normalized by its own kept-nontoxic flag rate
                if de <= 0 or dm <= 0:
                    return None
                return (fm[kt].mean() / dm) - (fe[kt].mean() / de)
            rm = block["methods"][m].get("selectivity_ratio")
            re_ = block["methods"]["encoder_e5"].get("selectivity_ratio")
            contrasts[f"{m}_minus_encoder"] = {
                "point": (round(rm - re_, 3) if rm is not None and re_ is not None else None),
                "ci95": clustered_boot(subs, diff)}
        block["llm_minus_encoder_ratio"] = contrasts
        res["by_tox_def"][dname] = block

    toxic_main = defs[f"Detoxify>={THRESH_MAIN}"]
    matched = {"target_removal_rates": {}, "note": "false-flag = P(score>=t_m | kept, toxic@0.5)"}
    # Human removal rate (fraction the community actually removed) anchors the first target so the
    # methods are compared at the operating point a real moderator would have chosen.
    human_rate = float((label == 1).mean())
    for r_target in (round(human_rate, 3), 0.40, 0.50):
        row = {}
        for m in methods:
            sc = scores[m]
            # Pick each method's own threshold so its overall removal rate equals the shared target;
            # this strips out aggressiveness differences and compares false-flags at equal strictness.
            t_m = np.quantile(sc, 1 - r_target)
            flag_m = (sc >= t_m).astype(float)
            kt = kept & toxic_main
            ff = float(flag_m[kt].mean()) if kt.sum() else None
            def ff_stat(ix, fm=flag_m):
                f, km, tmask = fm[ix], kept[ix], toxic_main[ix]
                kt = km & tmask
                return float(f[kt].mean()) if kt.sum() else None
            ci = clustered_boot(subs, ff_stat)
            row[m] = {"false_flag_kept_toxic": round(ff, 4) if ff is not None else None,
                      "achieved_removal_rate": round(float(flag_m.mean()), 4), "ci95": ci}
        matched["target_removal_rates"][str(r_target)] = row
    res["strictness_matched"] = matched

    det = tox_frame("Detoxify", base.select("subreddit", "idx")).sort("subreddit", "idx")
    tox = det["tox"].to_numpy()
    comm_rows = []
    for s in np.unique(subs):
        ix = np.where(subs == s)[0]
        toxic_here = ix[tox[ix] >= THRESH_MAIN]
        if len(toxic_here) >= 5:  # require a few toxic comments before a community's keep-rate is meaningful
            # High keep-rate among toxic = community empirically tolerates insults; these are exactly the
            # norms an over-aggressive model silences. Compare Gemma vs encoder flagging on that kept-toxic set.
            keep_rate = float((label[toxic_here] == 0).mean())
            kt_here = toxic_here[label[toxic_here] == 0]
            gflag = float(flags["llm_gemma"][kt_here].mean()) if len(kt_here) else None
            eflag = float(flags["encoder_e5"][kt_here].mean()) if len(kt_here) else None
            comm_rows.append({"subreddit": s, "n_toxic": int(len(toxic_here)),
                              "keep_rate_among_toxic": round(keep_rate, 3),
                              "n_kept_toxic": int(len(kt_here)),
                              "gemma_flags_kept_toxic": round(gflag, 3) if gflag is not None else None,
                              "encoder_flags_kept_toxic": round(eflag, 3) if eflag is not None else None})
    comm_rows.sort(key=lambda r: -r["keep_rate_among_toxic"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    OUT_COMM.write_text(json.dumps({"analysis": "insult_tolerant_communities",
                                    "definition": "communities with >=5 toxic comments, ranked by keep-rate among toxic",
                                    "top": comm_rows[:12]}, indent=2))

    print(f"n_common_rows={n_common}")
    for dname in (f"Detoxify>={THRESH_MAIN}", "Detoxify_top_decile_per_comm", "ToxiGen>=0.5"):
        b = res["by_tox_def"][dname]
        print(f"\n[{dname}] selectivity ratio P(flag|kept,tox)/P(flag|kept,nontox):")
        for m, v in b["methods"].items():
            if v.get("selectivity_ratio") is not None:
                print(f"   {m:11s} ratio={v['selectivity_ratio']:.2f} {v['ratio_ci95']} "
                      f"(flag_tox={v['p_flag_kept_toxic']:.3f} flag_nontox={v['p_flag_kept_nontoxic']:.3f} "
                      f"n_kt={v['n_kept_toxic']})")
    print("\n[strictness-matched @ human removal rate] false-flag on kept-toxic:")
    r0 = list(res["strictness_matched"]["target_removal_rates"].keys())[0]
    for m, v in res["strictness_matched"]["target_removal_rates"][r0].items():
        print(f"   {m:11s} false_flag={v['false_flag_kept_toxic']} {v['ci95']} (achieved r={v['achieved_removal_rate']})")
    print("\nTop empirical insult-tolerant communities (keep-rate among toxic):")
    for c in comm_rows[:5]:
        print(f"   r/{c['subreddit']:22s} keep@tox={c['keep_rate_among_toxic']:.2f} n_tox={c['n_toxic']} "
              f"gemma_flags_kept_tox={c['gemma_flags_kept_toxic']} enc={c['encoder_flags_kept_toxic']}")


if __name__ == "__main__":
    main()
