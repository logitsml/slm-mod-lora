"""Decision-axis community-invariance (rank 3) -- the strongest POSITIVE internals claim: at the
decision token gemma-3-12b-it carries ONE near-community-invariant remove-vs-keep axis, not 95.

Consumes the shared decision-token collection (decision_axis_collect.py). For each SAE layer we compute
the diff-of-means "remove" direction globally (d_global) and per community (d_s), then test whether the
per-community axes are collinear with each other and with d_global, AGAINST a within-community split-half
reliability CEILING (the most a per-community direction can correlate with itself given finite samples).

If cross-community cos(d_s, d_s') reaches >= ~0.85 of the within-community ceiling, and per-community
remove directions are far MORE collinear than the per-community community-IDENTITY directions (which are
distinct), then the model applies one universal removability axis even though it represents communities
distinctly -- the mechanistic WHY of the NAT-vs-BAL gap. We also report cos(d_global, d_rule_text)
to unify with the rule-presence lever (rule_presence collection arm). NOTE: the norules arm
swaps only the rule TEXT (the real community DESCRIPTION is retained in both arms), so this cosine
measures alignment with the RULE-TEXT direction, NOT removal of all community context.

INVARIANCE IS AN EQUIVALENCE CLAIM, NOT A FAILURE-TO-REJECT .
"One universal axis" is a statement of EQUIVALENCE (cross approx ceiling), so it must be established
with an equivalence / TOST framing -- NOT by failing to reject a difference test. We therefore claim
invariance ONLY when the clustered-bootstrap LOWER bound of cross_over_ceiling_ratio >= a
PRE-REGISTERED threshold INVARIANCE_RATIO_THR (0.85): the whole CI for the ratio lies in the
"practically equal" region. The paired within-vs-cross PERMUTATION p is retained ONLY as a difference
test (small p => within EXCEEDS cross => evidence AGAINST invariance); it is explicitly NOT evidence
FOR invariance (absence of evidence != evidence of equivalence). This is exactly the pathology a
difference-test framing invites and that the TOST/equivalence call avoids: we report a paired
permutation DIFFERENCE test alongside a TOST EQUIVALENCE test on the cross/ceiling ratio, and the
invariance claim rests on the equivalence test only.

GEMMA-SPECIFIC HANDLING: per-dim z-score + drop the top-1
massive-activation dim before all cosines; report the cross/within-ceiling ratio, never raw cosine vs 1.0.

STATS:
  (1) The cross-community direction is sample-size-matched to the split-half ceiling: both use a
      HALF-size-per-class diff-of-means, so the ceiling is not spuriously low (diff-of-means noise
      grows with smaller N) and the cross/ceiling ratio is not inflated.
  (2) Uncertainty + the equivalence decision come from a community-CLUSTERED block bootstrap
      (resample SUBREDDITS, ~2000x) -- the correct unit, since pairwise cross-cosines are dependent
      and the ceiling is one value per community. The TOST/equivalence call is made from the
      cross_over_ceiling_ratio CI lower bound (see the equivalence note above). The retained permutation p is a
      clean paired DIFFERENCE stat (per community: OWN split-half ceiling vs mean cross-cos to others,
      within/cross label permuted per community) -- reported as a difference test ONLY, never as proof
      of invariance.

  smoke uses whatever collection exists.
  run:  env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.decision_axis_invariance
Out: results/kumar_mod/decision_axis_invariance.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEC = ROOT / "results" / "kumar_mod" / "decisiontok"
OUT = ROOT / "results" / "kumar_mod" / "decision_axis_invariance.json"
LAYERS = [12, 24, 31, 41]
SEED = 11
# A community needs at least this many comments per class for its diff-of-means
# direction to be stable enough to enter the cross-community pool and the split-half ceiling.
MIN_PER_CLASS = 15
# Pre-registered TOST equivalence threshold: invariance is claimed only when the
# bootstrap LOWER bound of cross/ceiling clears this, never on a raw cosine vs 1.0.
INVARIANCE_RATIO_THR = 0.85


def _zspace(X):
    """Per-dim standardize + drop the single highest-variance (massive-activation) dim."""
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Z = (X - mu) / sd
    # Gemma carries one massive-activation channel that dominates raw variance and
    # would otherwise hijack every cosine; zero it out (the dim index is reported).
    top = int(np.argmax(X.var(0)))
    Z[:, top] = 0.0
    return Z, top


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _diff_means(Z, y):
    return Z[y == 1].mean(0) - Z[y == 0].mean(0)


def _half_subsample_dir(Zs, ys, rng):
    """Diff-of-means direction from a half-size-per-class subsample of one community.

    Sample-size matches the per-community 'cross' direction to the SAME per-class count
    used by the within-community split-half ceiling, so cross-community cosines and the
    reliability ceiling sit on the same sampling-noise footing; matching the per-class
    counts avoids a spuriously low ceiling that would inflate the cross/ceiling ratio.
    """
    pos = np.where(ys == 1)[0]; neg = np.where(ys == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    # Keep only half of each class so this direction is built from the same per-class
    # count as one split-half of the ceiling -- same sampling-noise footing.
    sel = np.concatenate([pos[: len(pos) // 2], neg[: len(neg) // 2]])
    yy = ys[sel]
    return Zs[sel][yy == 1].mean(0) - Zs[sel][yy == 0].mean(0)


def _pairwise_cos_subset(dir_map, names):
    """Upper-triangle pairwise cosines among a named subset of directions (clustered bootstrap)."""
    D = np.array([_unit(dir_map[s]) for s in names])
    C = D @ D.T
    iu = np.triu_indices(len(D), k=1)
    return C[iu]


def run():
    if not (DEC / "meta.parquet").exists():
        print(f"[invariance] no collection at {DEC} (run decision_axis_collect.py first)"); return
    meta = pl.read_parquet(DEC / "meta.parquet")
    import os as _os
    # Default axis contrasts the ground-truth remove/keep label. The env override instead
    # splits on the model's own logit-gap sign (gap_rules>0), used for model-decision variants.
    if _os.environ.get("DAI_MODEL_DECISION") and "gap_rules" in meta.columns:
        y = (meta["gap_rules"].to_numpy() > 0).astype(int)
    else:
        y = meta["label"].to_numpy().astype(int)
    sub = meta["subreddit"].to_numpy()
    subs = sorted(set(sub.tolist()))
    rng = np.random.default_rng(SEED)
    have_rule = (DEC / f"res_norules_L{LAYERS[0]}.fp16.npy").exists()

    out = {"analysis": "decision_axis_community_invariance", "model": "google/gemma-3-12b-it",
           "n_comments": int(meta.height), "n_communities_total": len(subs),
           "rule_presence_available": bool(have_rule), "by_layer": {}}

    for L in LAYERS:
        fp = DEC / f"res_rules_L{L}.fp16.npy"
        if not fp.exists():
            continue
        X = np.load(fp).astype(np.float32)
        Z, top_dim = _zspace(X)
        d_global = _diff_means(Z, y)


        d_s, d_s_half, ceil_by_s, id_by_s, used = {}, {}, {}, {}, []
        cent = Z.mean(0)
        for s in subs:
            m = sub == s
            ys = y[m]; Zs = Z[m]
            if (ys == 1).sum() < MIN_PER_CLASS or (ys == 0).sum() < MIN_PER_CLASS:
                continue
            d_s[s] = _diff_means(Zs, ys)
            # Community-IDENTITY direction: how this community's centroid sits off the
            # global centroid. The contrast that gives the result its content -- removability
            # axes can be collinear while these identity axes stay distinct.
            id_by_s[s] = Z[m].mean(0) - cent
            used.append(s)

            d_s_half[s] = _half_subsample_dir(Zs, ys, rng)

            # Within-community split-half reliability ceiling: split each class in two,
            # build a remove direction per half, cosine them. This is the most a finite-sample
            # per-community axis can correlate with a fresh estimate of ITSELF -- the bar the
            # cross-community cosines are judged against, not 1.0.
            pos = np.where(ys == 1)[0]; neg = np.where(ys == 0)[0]
            rng.shuffle(pos); rng.shuffle(neg)
            h1 = np.concatenate([pos[:len(pos)//2], neg[:len(neg)//2]])
            h2 = np.concatenate([pos[len(pos)//2:], neg[len(neg)//2:]])
            d1 = Zs[h1][ys[h1] == 1].mean(0) - Zs[h1][ys[h1] == 0].mean(0)
            d2 = Zs[h2][ys[h2] == 1].mean(0) - Zs[h2][ys[h2] == 0].mean(0)
            ceil_by_s[s] = float(_unit(d1) @ _unit(d2))

        if len(used) < 3:
            out["by_layer"][L] = {"note": "too few communities with enough per-class samples", "n_used": len(used)}
            continue

        ceil_cos = np.array([ceil_by_s[s] for s in used])

        # Cross-community collinearity uses the half-size directions so each cosine is
        # built on the same per-class N as one arm of the ceiling.
        cross_cos = _pairwise_cos_subset(d_s_half, used)
        to_global = np.array([_unit(d_s[s]) @ _unit(d_global) for s in used])
        within_ceiling = float(np.median(ceil_cos))
        cross_median = float(np.median(cross_cos))


        id_cos = _pairwise_cos_subset(id_by_s, used)
        id_median = float(np.median(id_cos))


        # Community-clustered block bootstrap: resample SUBREDDITS (not comments), since
        # the pairwise cross-cosines are dependent and the ceiling is one value per community.
        used_arr = np.array(used)
        n_used = len(used)
        bs_cross, bs_ceil, bs_gap, bs_ratio, bs_remov_minus_id = [], [], [], [], []
        for _ in range(2000):
            idx = rng.integers(0, n_used, size=n_used)
            samp = used_arr[idx]
            # Dedup the resample: a pairwise upper-triangle needs >=2 distinct communities,
            # and self-pairs (a duplicate against itself) would be spurious cos=1 entries.
            uniq = list(dict.fromkeys(samp.tolist()))
            if len(uniq) < 2:
                continue


            bc = float(np.median(np.array([ceil_by_s[s] for s in uniq])))
            xc = float(np.median(_pairwise_cos_subset(d_s_half, uniq)))
            ic = float(np.median(_pairwise_cos_subset(id_by_s, uniq)))
            bs_ceil.append(bc); bs_cross.append(xc)
            bs_gap.append(bc - xc)
            bs_ratio.append(xc / bc if bc > 0 else np.nan)
            # Removability-minus-identity: positive => remove axes are MORE shared across
            # communities than the communities' own identity axes are.
            bs_remov_minus_id.append(xc - ic)

        def _ci(a):
            a = np.array([v for v in a if np.isfinite(v)])
            if a.size == 0:
                return [None, None]
            return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


        # Paired difference stat, one pair per community: its OWN split-half ceiling (within)
        # vs its mean cosine to every other community's remove axis (cross). Retained as a
        # difference test ONLY -- small p => within exceeds cross => evidence AGAINST invariance.
        Dhalf = {s: _unit(d_s_half[s]) for s in used}
        mean_cross_to_others = {}
        for s in used:
            others = [Dhalf[s] @ Dhalf[o] for o in used if o != s]
            mean_cross_to_others[s] = float(np.mean(others)) if others else np.nan
        paired_within = np.array([ceil_by_s[s] for s in used])
        paired_cross = np.array([mean_cross_to_others[s] for s in used])
        valid = np.isfinite(paired_cross)
        paired_within = paired_within[valid]; paired_cross = paired_cross[valid]
        obs_paired = float(np.median(paired_within) - np.median(paired_cross))
        n_p = len(paired_within)
        perm_diffs = []
        stacked = np.stack([paired_within, paired_cross], axis=1)
        for _ in range(2000):
            # Permute the within/cross label within each community (random sign flip),
            # the exchangeable unit under the null of no within-vs-cross difference.
            flip = rng.integers(0, 2, size=n_p)
            a = np.where(flip == 0, stacked[:, 0], stacked[:, 1])
            b = np.where(flip == 0, stacked[:, 1], stacked[:, 0])
            perm_diffs.append(float(np.median(a) - np.median(b)))
        # One-sided (within >= cross), +1 in num and denom for the observed value.
        pval = float((np.sum(np.array(perm_diffs) >= obs_paired) + 1) / (len(perm_diffs) + 1))


        # TOST call: the whole ratio CI must sit in the practically-equal region, so the
        # decision keys on the CI LOWER bound, not the point estimate.
        ratio_ci = _ci(bs_ratio)
        ratio_ci_lo = ratio_ci[0]
        invariance_supported = bool(ratio_ci_lo is not None and ratio_ci_lo >= INVARIANCE_RATIO_THR)

        layer_res = {
            "n_used_communities": len(used),
            "cross_community_remove_cos_median": round(cross_median, 4),
            "cross_community_remove_cos_ci95": _ci(bs_cross),
            "within_community_split_half_ceiling": round(within_ceiling, 4),
            "within_community_split_half_ceiling_ci95": _ci(bs_ceil),
            "cross_over_ceiling_ratio": round(cross_median / within_ceiling, 4) if within_ceiling > 0 else None,
            "cross_over_ceiling_ratio_ci95": ratio_ci,
            "ceiling_minus_cross_gap_ci95": _ci(bs_gap),
            "median_cos_to_global_remove": round(float(np.median(to_global)), 4),
            "community_identity_pairwise_cos_median": round(id_median, 4),
            "removability_minus_identity_collinearity": round(cross_median - id_median, 4),
            "removability_minus_identity_collinearity_ci95": _ci(bs_remov_minus_id),

            "invariance_ratio_threshold": INVARIANCE_RATIO_THR,
            "invariance_supported_tost": invariance_supported,


            "perm_pval_cross_vs_ceiling_paired_DIFFERENCE_TEST": round(pval, 4),
            "stat_method": "community-clustered block bootstrap (resample subreddits, 2000x); "
                           "INVARIANCE = TOST: ratio CI lower bound >= pre-registered threshold "
                           f"({INVARIANCE_RATIO_THR}); paired-by-community within-vs-cross label "
                           "permutation reported as a DIFFERENCE test only (not invariance evidence); "
                           "cross-cos uses half-size-per-class directions sample-size-matched to the ceiling",
            "massive_activation_dim_dropped": top_dim,
        }
        if have_rule:
            Xn = np.load(DEC / f"res_norules_L{L}.fp16.npy").astype(np.float32)


            norules_top = int(np.argmax(Xn.var(0)))
            assert norules_top == top_dim, (
                f"norules massive-activation dim {norules_top} != rules top_dim {top_dim} at L{L}; "
                "the two arms disagree on the highest-variance channel -- do not trust the rule-presence "
                "cosine (would zero the wrong dim).")
            # Standardize the norules arm with the RULES arm's mu/sd (same frame) so the
            # per-comment difference isolates the rule-text effect, then zero the shared
            # massive-activation dim. d_rule = mean shift toward rule-text presence (rules minus norules).
            Zn = (Xn - X.mean(0)) / (X.std(0) + 1e-6); Zn[:, top_dim] = 0.0
            d_rule = (Z - Zn).mean(0)


            layer_res["cos_global_remove_vs_rule_text_presence"] = round(float(_unit(d_global) @ _unit(d_rule)), 4)
        out["by_layer"][L] = layer_res
        print(f"[invariance L{L}] cross-cos {cross_median:.3f} / ceiling {within_ceiling:.3f} "
              f"= {layer_res['cross_over_ceiling_ratio']} (ratio CI95 {layer_res['cross_over_ceiling_ratio_ci95']}) "
              f"| TOST invariance_supported={invariance_supported} (thr {INVARIANCE_RATIO_THR}) "
              f"| identity-cos {layer_res['community_identity_pairwise_cos_median']} "
              f"| paired DIFFERENCE-test p={pval:.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[invariance] SAVED -> {OUT}", flush=True)
    return out


if __name__ == "__main__":
    run()
