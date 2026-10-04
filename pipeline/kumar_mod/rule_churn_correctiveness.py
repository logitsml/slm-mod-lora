
"""Rule-churn correctiveness. Produces analysis/rule_churn_correctiveness.json.

Among paired rows whose rule intervention changes the binary decision (cond home -> scrambled / none /
random), delta_correct = Pr(intervention decision == recorded label) - Pr(home decision == recorded label).
Negative means churn moves away from the recorded label more often than toward it. Reads
results/kumar_mod/rule_scramble_{gemma,llama,qwen}.parquet; unparsed decisions (null would_moderate) are
dropped before pairing, so n_paired varies slightly by condition.

Bootstrap CIs are subreddit-clustered percentile CIs (2000 reps, a single shared seed (11)).

CPU only. Out: results/kumar_mod/analysis/rule_churn_correctiveness.json
(override with --out for verification runs).
"""
import argparse, json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = str(Path(__file__).resolve().parents[2]) + ""
R = f"{ROOT}/results/kumar_mod"
FAMS = ["gemma", "llama", "qwen"]
# Each cond is paired against the "home" (real-rules) decision for the same row.
CONDS = ["scrambled", "none", "random"]
SEED = 11        # shared project seed; fixes the bootstrap resampling
NBOOT = 2000


def load_pairs(fam, cond):
    # non-strict cast leaves unparsed would_moderate as null; the is_in filter then drops them,
    # so only rows with a clean binary decision survive (n_paired varies by cond as a result).
    d = pl.read_parquet(f"{R}/rule_scramble_{fam}.parquet").with_columns(
        pl.col("would_moderate").cast(pl.Int8, strict=False).alias("wm")
    ).filter(pl.col("wm").is_in([0, 1]))
    home = d.filter(pl.col("cond") == "home").select(["subreddit", "idx", "label", pl.col("wm").alias("wm0")])
    it = d.filter(pl.col("cond") == cond).select(["subreddit", "idx", pl.col("wm").alias("wm1")])
    # inner join on (subreddit, idx) pairs the same item under home vs. intervention; a row counts
    # only if it has a parsed decision in both conditions.
    return home.join(it, on=["subreddit", "idx"], how="inner")


def cluster_ci(delta_flip, clusters, seed):
    """Percentile CI of delta_correct from a cluster bootstrap over the flip rows.
    delta_flip: per-flip +1 (moved toward label) / -1 (moved away). clusters: cluster key per flip."""
    uniq = sorted(set(clusters))
    byc = {c: np.where(np.asarray(clusters) == c)[0] for c in uniq}  # cluster -> its flip-row indices
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(NBOOT):
        # resample whole clusters with replacement, then pool all rows in the drawn clusters;
        # clustering on subreddit keeps within-community correlation from shrinking the CI.
        samp = [uniq[i] for i in rng.integers(0, len(uniq), len(uniq))]
        ix = np.concatenate([byc[c] for c in samp])
        boots.append(float(delta_flip[ix].mean()))
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def block(flips, ci):
    # flips are strictly +-1, so among rows that changed decision exactly one side is right:
    # toward/n is the intervention's accuracy, away/n the home's, and delta_correct their difference.
    n = len(flips)
    toward = int((flips == 1).sum()); away = int((flips == -1).sum())
    return {"n_flips": n,
            "intervention_correct_given_flip": round(toward / n, 4),
            "base_correct_given_flip": round(away / n, 4),
            "delta_correct": round(toward / n - away / n, 4)}, toward, away, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/analysis/rule_churn_correctiveness.json")
    args = ap.parse_args()

    by_family = {}
    all_delta, all_sub, all_fcs = [], [], []
    for fam in FAMS:
        by_family[fam] = {}
        for cond in CONDS:
            p = load_pairs(fam, cond)
            f = p.filter(pl.col("wm0") != pl.col("wm1"))  # "flips": rows where the intervention changed the decision

            # +1 if the intervention decision matches the recorded label (moved toward it), -1 otherwise.
            # On a flip the home decision is the complement, so -1 means home was the correct side.
            delta_flip = np.where(f["wm1"].to_numpy() == f["label"].to_numpy(), 1.0, -1.0)
            subs = f["subreddit"].to_list()
            b, toward, away, ci = block(delta_flip, cluster_ci(delta_flip, subs, SEED))
            by_family[fam][cond] = {
                "n_paired": p.height, "n_flips": b["n_flips"],
                "flip_rate": round(b["n_flips"] / p.height, 4),
                "intervention_correct_given_flip": b["intervention_correct_given_flip"],
                "base_correct_given_flip": b["base_correct_given_flip"],
                "delta_correct": b["delta_correct"],
                "ci95_cluster_subreddit": ci,
                "moves_toward_label_n": toward, "moves_away_from_label_n": away,
                # directional breakdown of the flips: home-removes -> intervention-keeps, and vice versa
                "removal_to_keep_flips": f.filter((pl.col("wm0") == 1) & (pl.col("wm1") == 0)).height,
                "keep_to_removal_flips": f.filter((pl.col("wm0") == 0) & (pl.col("wm1") == 1)).height,
                "condition_label": {"random": "neutral_words"}.get(cond, cond),  # "random" rule = neutral-word placeholder
            }
            all_delta.append(delta_flip)
            all_sub.extend(subs)
            # keys for the pooled CI clustered at family|cond|subreddit, so the same subreddit
            # under different (fam, cond) is treated as a distinct cluster.
            all_fcs.extend([f"{fam}|{cond}|{s}" for s in subs])

    pooled = np.concatenate(all_delta)
    n = len(pooled); toward = int((pooled == 1).sum()); away = int((pooled == -1).sum())
    base = {"n_flips": n,
            "intervention_correct_given_flip": round(toward / n, 4),
            "base_correct_given_flip": round(away / n, 4),
            "delta_correct": round(toward / n - away / n, 4)}
    out = {
        "analysis": "rule_churn_correctiveness",
        "definition": ("Among paired rows whose rule intervention changes the binary decision, delta_correct = "
                       "Pr(intervention decision equals recorded label) - Pr(home decision equals recorded label). "
                       "Negative means churn moves away from the recorded label more often than toward it."),
        "by_family": by_family,
        # Two pooled CIs over the same point estimate, differing only in cluster granularity:
        # fine (family|cond|subreddit) vs. coarse (subreddit collapses across fam/cond).
        "pooled_family_condition_subreddit_cluster": dict(base, **{
            "ci95_cluster_family_condition_subreddit": cluster_ci(pooled, all_fcs, SEED)}),
        "pooled_subreddit_cluster": dict(base, **{
            "ci95_cluster_subreddit": cluster_ci(pooled, all_sub, SEED)}),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)
    print("pooled:", json.dumps(base))


if __name__ == "__main__":
    main()
