"""Toxicity-collapse invariance across prompt conditions.

For every prompt condition that produces per-comment decisions on the SHARED test-fold rows -- the anti-tox
paraphrases (run_antitox: baseline, p1-p4) and the rule-manipulations (rule_scramble: home, scrambled, random,
none) -- join the toxicity scores (Detoxify + s-nlp + ToxiGen + lexical) on (subreddit, idx) and compute, per
(family, condition):
  - auc_tox_predicts_decision : roc_auc_score(would_moderate, tox_score) per scorer -- how strongly toxicity
                                predicts the LLM's removal. HIGH + FLAT across conditions => collapse is
                                prompt-invariant (rules / anti-tox instructions don't move it).
  - nontox_balacc            : balanced_accuracy(true_label, decision) on the NON-toxic subset (tox<0.1) --
                                does the model discriminate removed-vs-kept among non-toxic comments. ~0.5 =>
                                blind to non-toxic violations.
Reports the RANGE across the anti-tox paraphrases (the key "can't be prompted away" number) and across all
conditions. NOTE on coverage: banana(absurd) + framing + other-subs (community_context_swap) are RATE-based
conditions on their own samples (not (subreddit,idx)-joinable), so they are reported separately; this table
is the toxicity-AUC view over the idx-joinable conditions.

Out: results/kumar_mod/analysis/toxicity_across_conditions.json
Run: env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.toxicity_across_conditions
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import polars as pl
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

KM = ROOT / "results" / "kumar_mod"
QW = KM / "analysis"
DETOX = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
MULTITOX = ROOT / "data" / "processed" / "kumar_balanced_multitox.parquet"
FAMS = ["gemma", "llama", "qwen", "gemma4_12b", "qwen36_27b"]
# "non-toxic" cutoff on the primary (Detoxify) score; below this a comment is treated as clearly clean,
# so removals here can't be explained away as toxicity-tracking.
NONTOX_THR = 0.1


def _tox_table():
    """(subreddit, idx) -> the available toxicity scores, joined."""
    t = None
    if DETOX.exists():
        t = pl.read_parquet(DETOX, columns=["subreddit", "idx", "tox_toxicity"]).rename(
            {"tox_toxicity": "tox_detoxify"})
    if MULTITOX.exists():
        m = pl.read_parquet(MULTITOX)
        # pull every tox_* column (s-nlp, ToxiGen, lexical) so AUC can be checked across all scorers
        keep = ["subreddit", "idx"] + [c for c in m.columns if c.startswith("tox_")]
        m = m.select(keep)
        # inner join keeps only rows scored by both files, so every scorer covers the same comment set
        t = m if t is None else t.join(m, on=["subreddit", "idx"], how="inner")
    return t


def _cond_rows(fam):
    """Yield (source, cond, DataFrame[subreddit, idx, label, would_moderate]) for the idx-joinable conditions."""
    for src, fn in (("antitox", KM / f"antitox_{fam}.parquet"),
                    ("rule_scramble", KM / f"rule_scramble_{fam}.parquet")):
        if not fn.exists():
            continue
        d = pl.read_parquet(fn)
        if "cond" not in d.columns:
            continue
        # one (family, condition) cell per prompt variant within the file (e.g. baseline, p1-p4 / home, scrambled...)
        for cond in d["cond"].unique().to_list():
            g = d.filter(pl.col("cond") == cond).select(["subreddit", "idx", "label", "would_moderate"])
            yield src, cond, g


def _scorer_cols(tox):
    return [c for c in tox.columns if c.startswith("tox_")]


def main():
    QW.mkdir(parents=True, exist_ok=True)
    tox = _tox_table()
    if tox is None:
        (QW / "toxicity_across_conditions.json").write_text(json.dumps({"status": "no toxicity parquets yet"}))
        print("no toxicity parquets"); return
    scorers = _scorer_cols(tox)
    out = {"analysis": "toxicity_across_conditions", "nontox_threshold": NONTOX_THR,
           "scorers": scorers, "by_family": {}}
    for fam in FAMS:
        fam_rows = list(_cond_rows(fam))
        if not fam_rows:
            continue
        out["by_family"][fam] = {}
        for src, cond, g in fam_rows:
            # drop unparsed decisions before joining; inner join attaches the toxicity scores on (subreddit, idx)
            j = g.drop_nulls(["would_moderate"]).join(tox, on=["subreddit", "idx"], how="inner")
            if j.height == 0:
                continue
            dec = j["would_moderate"].to_numpy().astype(int)
            lab = j["label"].to_numpy().astype(int)
            row = {"n": int(j.height), "removal_rate": round(float(dec.mean()), 4),
                   "auc_tox_predicts_decision": {}}
            for sc in scorers:
                s = j[sc].to_numpy().astype(float)
                ok = np.isfinite(s)
                # AUC needs both decision classes present and enough rows; otherwise it's undefined -> None
                if len(np.unique(dec[ok])) == 2 and ok.sum() > 10:
                    # how well this scorer's toxicity ranks the LLM's removal decisions
                    row["auc_tox_predicts_decision"][sc] = round(float(roc_auc_score(dec[ok], s[ok])), 4)
                else:
                    row["auc_tox_predicts_decision"][sc] = None

            # restrict to clearly non-toxic comments (prefer Detoxify) and ask whether the model still tells
            # removed from kept there; balanced_accuracy ~0.5 means it's blind to non-toxic violations
            tcol = "tox_detoxify" if "tox_detoxify" in j.columns else scorers[0]
            tv = j[tcol].to_numpy().astype(float)
            nt = np.isfinite(tv) & (tv < NONTOX_THR)
            # need both true classes in the subset; balanced_accuracy guards against the skewed non-tox base rate
            if nt.sum() > 10 and len(np.unique(lab[nt])) == 2 and len(np.unique(dec[nt])) >= 1:
                row["nontox_balacc"] = round(float(balanced_accuracy_score(lab[nt], dec[nt])), 4)
                row["nontox_n"] = int(nt.sum())
            else:
                row["nontox_balacc"] = None
            out["by_family"][fam][f"{src}:{cond}"] = row

        fa = out["by_family"][fam]
        # summarize over the anti-tox arm only, on one scorer (Detoxify when present) for a single headline number
        prim = scorers[0] if "tox_detoxify" not in scorers else "tox_detoxify"
        # the dosed paraphrases (p1-p4); baseline is the no-instruction reference compared against
        anti = {k: v for k, v in fa.items() if k.startswith("antitox:") and k != "antitox:baseline"}
        base = fa.get("antitox:baseline", {})
        base_auc = (base.get("auc_tox_predicts_decision") or {}).get(prim)
        anti_aucs = [v["auc_tox_predicts_decision"].get(prim) for v in anti.values()
                     if v.get("auc_tox_predicts_decision", {}).get(prim) is not None]
        if base_auc is not None and anti_aucs:
            fa["_antitox_summary"] = {
                "scorer": prim, "baseline_auc_tox": base_auc,
                "antitox_auc_tox_min": round(min(anti_aucs), 4), "antitox_auc_tox_max": round(max(anti_aucs), 4),
                "max_drop_from_baseline": round(base_auc - min(anti_aucs), 4),
                # collapse is "not promptable-away" only if anti-tox instructions barely dent the AUC: the
                # best paraphrase stays within 0.03 of baseline and even the worst stays within 0.05
                "verdict": ("toxicity-tracking NOT reduced by anti-tox prompting (collapse not promptable-away)"
                            if base_auc - max(anti_aucs, default=base_auc) < 0.03 and base_auc - min(anti_aucs) < 0.05
                            else "anti-tox prompting moved toxicity-tracking -- inspect")}
    (QW / "toxicity_across_conditions.json").write_text(json.dumps(out, indent=2, default=str))
    nfam = len(out["by_family"])
    print(f"[tox-across-conditions] {nfam} families, scorers={scorers} -> {QW/'toxicity_across_conditions.json'}")
    for fam, fa in out["by_family"].items():
        s = fa.get("_antitox_summary")
        if s:
            print(f"  {fam}: baseline tox-AUC {s['baseline_auc_tox']} vs anti-tox "
                  f"[{s['antitox_auc_tox_min']}, {s['antitox_auc_tox_max']}] -> {s['verdict']}")


if __name__ == "__main__":
    main()
