"""Aggregate the scrambled-rules condition (rule_scramble_run) across families. The headline test: is the
SCRAMBLED-rules removal rate ~= the HOME (real-rules) rate? If yes, the model reacts to rule PRESENCE, not
content -- and because scrambled == home in length/tokens, that conclusion is free of the prompt-length
confound the foreign-rules condition could not rule out.

Per family + condition (home / scrambled / random / none): removal rate, toxicity-tracking AUC(tox->remove), and the
paired decision-flip rate vs home. CPU. Out: results/kumar_mod/analysis/rule_scramble.json
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import polars as pl
from sklearn.metrics import roc_auc_score

OUT = ROOT / "results" / "kumar_mod"
QW = OUT / "analysis"
QW.mkdir(parents=True, exist_ok=True)
TOX = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
FAMS = ["gemma", "llama", "qwen"]
# home = real subreddit rules; scrambled = same rules with words shuffled to nonsense (length-matched
# absurd-rules control); random = same rules with every word replaced by a neutral filler word (keywords
# gone, form preserved); none = no rules block.
CONDS = ["home", "scrambled", "random", "none"]


def main():
    tox = pl.read_parquet(TOX, columns=["subreddit", "idx", "tox_toxicity"]) if TOX.exists() else None
    res = {"analysis": "rule_scramble",
           "design": "Kumar prompt under home / scrambled (same rules, words shuffled to nonsense) / "
                     "random (rules with every word replaced by neutral filler) / none, per family; "
                     "scrambled is a length-matched absurd-rules control.",
           "by_family": {}}
    for fam in FAMS:
        p = OUT / f"rule_scramble_{fam}.parquet"
        if not p.exists():
            res["by_family"][fam] = {"status": "not run yet"}
            continue
        # drop rows where the decision failed to parse; one comment carries up to four cond rows
        d = pl.read_parquet(p).filter(pl.col("would_moderate").is_not_null())
        if tox is not None:
            # attach the Detoxify toxicity score per (subreddit, comment) for the tracking-AUC below
            d = d.join(tox, on=["subreddit", "idx"], how="left")
        per = {}
        for c in CONDS:
            g = d.filter(pl.col("cond") == c)
            if g.height == 0:
                continue
            wm = g["would_moderate"].to_numpy().astype(int)
            entry = {"n": int(g.height), "removal_rate": round(float(wm.mean()), 4)}
            if tox is not None:
                gt = g.filter(pl.col("tox_toxicity").is_not_null())
                y = gt["would_moderate"].to_numpy().astype(int); tx = gt["tox_toxicity"].to_numpy()
                # how well toxicity ranks the removals within this condition; needs both classes present
                entry["auc_toxicity_predicts_remove"] = (round(float(roc_auc_score(y, tx)), 4)
                                                         if len(np.unique(y)) > 1 else None)
            per[c] = entry

        # one row per comment, one column per condition -> paired decisions on the same comment
        w = d.pivot(values="would_moderate", index=["subreddit", "idx"], on="cond")
        flips = {}
        for c in ("scrambled", "random", "none"):
            if c in w.columns and "home" in w.columns:
                # paired: only comments decided under both arms, so a flip is a true within-comment change
                x = w.drop_nulls(["home", c])
                flips[f"home_vs_{c}_flip_rate"] = round(float((x["home"] != x[c]).mean()), 4) if x.height else None
        hr = per.get("home", {}).get("removal_rate")
        sr = per.get("scrambled", {}).get("removal_rate")
        nr = per.get("none", {}).get("removal_rate")
        res["by_family"][fam] = {
            "by_condition": per, "flip_vs_home": flips,
            # headline gaps in percentage points: scrambled-minus-home tests whether rule content matters
            # (closer to 0 = more inert); none below home isolates how much removal the rules block alone drives
            "scrambled_minus_home_removal_pp": (round((sr - hr) * 100, 2) if (sr is not None and hr is not None) else None),
            "none_minus_home_removal_pp": (round((nr - hr) * 100, 2) if (nr is not None and hr is not None) else None),
        }

    lines = []
    for fam in FAMS:
        f = res["by_family"].get(fam, {})
        if "by_condition" in f:
            bc = f["by_condition"]
            lines.append(f"{fam}: removal home {bc.get('home', {}).get('removal_rate')} / scrambled "
                         f"{bc.get('scrambled', {}).get('removal_rate')} / random "
                         f"{bc.get('random', {}).get('removal_rate')} / none {bc.get('none', {}).get('removal_rate')}; "
                         f"home-vs-random flip {f.get('flip_vs_home', {}).get('home_vs_random_flip_rate')}")
    res["interpretation"] = (
        "If scrambled removal ~= home removal (and the home-vs-scrambled flip rate is low), the model reacts to "
        "the PRESENCE of rule text, not its content -- the 'rules' are a strictness dial, not the community's "
        "norm. Scrambled is length-matched to home, so this is free of the prompt-length confound. Per family: "
        + " | ".join(lines))
    (QW / "rule_scramble.json").write_text(json.dumps(res, indent=2))
    print("[rule_scramble] " + res["interpretation"])


if __name__ == "__main__":
    main()
