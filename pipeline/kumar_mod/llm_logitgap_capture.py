"""Capture each prompted LLM's next-token yes/no LOGIT GAP on the 80/20 test fold (parallel to how SLM-Mod is
scored), then compute the within-community ROC-AUC / PR-AUC from the gap. Model set via env DAI_MODEL; reuses
the Kumar prompt + decision-token logic from decision_axis_collect. argmax_ok reports the fraction of comments
where the model's top token is actually 'yes' or 'no' (sanity-check on the token ids for that tokenizer).

  CUDA_VISIBLE_DEVICES=N DAI_MODEL=meta-llama/Llama-3.1-8B-Instruct \
     python -m pipeline.kumar_mod.llm_logitgap_capture --tag llama31_8b [--smoke]
Out: results/kumar_mod/llm_gap_{tag}.parquet  and  llm_gap_{tag}.json
"""
from __future__ import annotations
import os
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.decision_axis_collect import _load_model, _build_prompt, MAX_LEN


def run(tag, smoke=False):
    import torch
    # Same seed-11 80/20 split as every other arm; score only the held-out test fold.
    split = pl.read_parquet(ROOT / "results/kumar_mod/balanced/slm_mod_split.parquet").filter(
        pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    tok, model = _load_model()
    dev = next(model.parameters()).device
    # Left-truncate so the JSON answer prefix at the tail (the decision token) always survives.
    tok.truncation_side = "left"
    # First sub-token of "yes"/"no"; the gap between these two logits is the decision signal.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    rows = split.to_dicts()
    if smoke:
        rows = rows[:12]
    bodies, out, argmax_hits = {}, [], 0
    for ci, r in enumerate(rows):
        s = r["subreddit"]
        if s not in bodies:
            bodies[s] = K.load_comments(s)
        # load_comments returns (body, label) tuples in Kumar's deduped first-seen order;
        # idx indexes into that list, [0] takes the body. Must match the split's indexing.
        body = bodies[s][r["idx"]][0]
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        # add_special_tokens=False: chat template already inserted <bos>; re-adding shifts positions.
        enc = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            # Logits at the final position: distribution over the next (decision) token.
            ll = model(**enc).logits[0, -1, :].to(torch.float32)
        # Sanity check: did the model actually want to emit yes/no here, not some other token?
        argmax_hits += int(int(ll.argmax().item()) in (yes_id, no_id))
        # gap = logit(yes) - logit(no): a continuous moderation score, ranked per community.
        out.append({"subreddit": s, "idx": int(r["idx"]), "label": int(r["label"]),
                    "gap": float((ll[yes_id] - ll[no_id]).item())})
        if smoke or (ci + 1) % 500 == 0:
            print(f"[gap-{tag}] {ci+1}/{len(rows)} argmax_ok={argmax_hits/(ci+1):.3f}", flush=True)
    df = pl.DataFrame(out)
    df.write_parquet(ROOT / f"results/kumar_mod/llm_gap_{tag}.parquet")
    # AUC is computed within each community separately, then aggregated -- a global AUC would
    # mostly reflect cross-community prevalence differences rather than discrimination.
    aucs, prs = [], []
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        y = g["label"].to_numpy().astype(int); sc = g["gap"].to_numpy().astype(float)
        # Skip communities AUC can't be defined on: one-class, too few rows, or non-finite gaps.
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10 or not np.isfinite(sc).all():
            continue
        aucs.append(roc_auc_score(y, sc)); prs.append(average_precision_score(y, sc))
    # Report the median over communities (robust to the long tail of small/skewed subreddits).
    summ = {"tag": tag, "model": os.environ.get("DAI_MODEL"), "n": df.height, "n_subs": len(aucs),
            "argmax_ok": round(argmax_hits / max(len(rows), 1), 4),
            "bal_auc_logitgap_median": round(float(np.median(aucs)), 4) if aucs else None,
            "pr_auc_logitgap_median": round(float(np.median(prs)), 4) if prs else None,
            "note": "within-community ROC-AUC of the yes/no logit gap predicting the moderator label; test fold"}
    json.dump(summ, open(ROOT / f"results/kumar_mod/llm_gap_{tag}.json", "w"), indent=2)
    print(f"[gap] {summ}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.tag, smoke=a.smoke)


if __name__ == "__main__":
    main()
