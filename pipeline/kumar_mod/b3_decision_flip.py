
"""B3 on ONE scale. Express Δ_rules and Δ_tox as the SAME quantity -- the fraction of the model's baseline
'remove' decisions that flip to 'keep' under an intervention -- so the inequality Δ_rules ≪ Δ_tox is
dimensionally clean.

  do(rules):  rule_scramble_{fam}.parquet, baseline cond='home' vs scrambled / none / random (random = every
              rule word replaced with a neutral filler word, structure kept; see rule_scramble_run.py)
  do(prompt): antitox_{fam}.parquet,       baseline vs strongest anti-tox prompt (p4)
  do(toxicity): the SAE necessity ablation on gemma-3-12b-it = 1.0 (sae_causal_dosedense.json), for reference

flip-to-keep rate is computed on baseline-removed comments (wm_home==1 -> wm_intervention==0), pooled,
with a subreddit-clustered bootstrap 95% CI. CPU only.
Out: results/kumar_mod/b3_decision_flip.json
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = str(Path(__file__).resolve().parents[2]) + ""
R = f"{ROOT}/results/kumar_mod"
SEED = 11
# White-box reference for Δ_tox: SAE necessity ablation on gemma-3-12b-it removes the
# toxicity feature and 100% of baseline-removed decisions flip to keep (sae_causal_dosedense.json).
SAE_TOX_FLIP = 1.0

def pivot_wm(path, base, others):
    # Keep only clean 0/1 moderation decisions; drop parse failures / abstains.
    d = pl.read_parquet(path).with_columns(pl.col("would_moderate").cast(pl.Int8).alias("wm")).filter(
        pl.col("wm").is_in([0, 1]))
    wide = None
    for c in [base] + others:
        s = d.filter(pl.col("cond") == c).select(["subreddit", "idx", "wm"]).rename({"wm": f"wm_{c}"})
        # Inner join on (subreddit, idx): a comment counts only if every condition decided it,
        # so flip rates are computed on the same paired set across all arms.
        wide = s if wide is None else wide.join(s, on=["subreddit", "idx"], how="inner")
    return wide

def flip_to_keep(wide, base, cond, seed=SEED + 1):
    # Denominator is the baseline-removed set only (wm_base==1); numerator is those that
    # become keep under the intervention. This makes do(rules) and do(prompt) the same scale.
    rem = wide.filter(pl.col(f"wm_{base}") == 1)
    flips = (rem[f"wm_{cond}"] == 0).to_numpy().astype(float)
    subs = rem["subreddit"].unique().to_list()
    # Precompute row positions per subreddit so the bootstrap can resample whole clusters cheaply.
    idxby = {s: np.where(rem["subreddit"].to_numpy() == s)[0] for s in subs}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(2000):
        # Subreddit-clustered resample: decisions within a community are correlated, so we
        # resample subreddits (with replacement) rather than individual comments.
        samp = [subs[i] for i in rng.integers(0, len(subs), len(subs))]
        ix = np.concatenate([idxby[s] for s in samp])
        boots.append(float(flips[ix].mean()))
    # Net removal-rate shift uses the full paired set (not just baseline-removed) so it can go
    # either direction; reported alongside the one-sided flip rate as a sanity cross-check.
    both = wide.filter(pl.col(f"wm_{base}").is_in([0, 1]) & pl.col(f"wm_{cond}").is_in([0, 1]))
    return {"flip_to_keep_rate": round(float(flips.mean()), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "n_baseline_removed": int(rem.height),
            "removal_rate_change_pp": round(float((both[f"wm_{cond}"].mean() - both[f"wm_{base}"].mean()) * 100), 2)}

out = {"analysis": "b3_same_scale_decision_flip", "scale": "fraction of baseline-removed decisions that flip to keep",
       "sae_tox_flip_gemma3 (Δ_tox)": SAE_TOX_FLIP, "by_family": {}}
for fam in ["gemma", "llama", "qwen"]:
    fam_res = {"do_rules": {}, "do_prompt": {}}
    # do(rules): baseline is the comment's home-subreddit rules; the three perturbations
    # (word-scrambled, no rules, neutral-filler random) all remove the real norm signal.
    rs = pivot_wm(f"{R}/rule_scramble_{fam}.parquet", "home", ["scrambled", "none", "random"])
    for c in ["scrambled", "none", "random"]:
        fam_res["do_rules"][c] = flip_to_keep(rs, "home", c)
    # do(prompt): baseline vs the strongest anti-toxicity instruction (p4) — Δ_tox proxy.
    at = pivot_wm(f"{R}/antitox_{fam}.parquet", "baseline", ["p4"])
    fam_res["do_prompt"]["anti_tox_p4"] = flip_to_keep(at, "baseline", "p4")
    out["by_family"][fam] = fam_res
    r = fam_res["do_rules"]; pr = fam_res["do_prompt"]
    print(f"[B3] {fam}: do(rules) scrambled={r['scrambled']['flip_to_keep_rate']} none={r['none']['flip_to_keep_rate']} "
          f"random={r['random']['flip_to_keep_rate']} | do(anti-tox)={pr['anti_tox_p4']['flip_to_keep_rate']} "
          f"| Δ_tox(SAE)={SAE_TOX_FLIP}  [removal change pp: scrambled={r['scrambled']['removal_rate_change_pp']}]",
          flush=True)
json.dump(out, open(f"{R}/b3_decision_flip.json", "w"), indent=2)
print("saved b3_decision_flip.json", flush=True)
