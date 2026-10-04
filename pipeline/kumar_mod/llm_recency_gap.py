"""Multi-card full-corpus yes/no LOGIT-GAP capture for the large recency models (Qwen3.6-27B needs 2 cards,
Llama-3.1-70B needs 4). Identical scoring path to llm_gemma4_gap.py (same _build_prompt, MAX_LEN, and
gap = logit[yes]-logit[no] read at the final position), so the resulting BAL/PR-AUC are directly comparable
to the gemma3/llama31/qwen25/gemma4 gap captures in Table 1. Only the model LOAD differs: device_map=auto
shards the model across all visible cards (accelerate). Runs the SAME full 80/20 test fold (no subsample).

  CUDA_VISIBLE_DEVICES=0,1[,2,3] python -m pipeline.kumar_mod.llm_recency_gap \
     --tag qwen36_27b --model Qwen/Qwen3.6-27B [--smoke]
Out: results/kumar_mod/llm_gap_{tag}.parquet  and  llm_gap_{tag}.json  (same names/schema as the other gaps)
"""
from __future__ import annotations
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.decision_axis_collect import _build_prompt, MAX_LEN


def run(tag, model_id, smoke=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    split = pl.read_parquet(ROOT / "results/kumar_mod/balanced/slm_mod_split.parquet").filter(
        pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    tok = AutoTokenizer.from_pretrained(model_id)

    # device_map=auto is the only departure from the single-card gemma4 path: accelerate shards
    # the model across every visible GPU so the 27B/70B weights fit. Math is unchanged.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    # inputs must land on the same card that holds the embedding layer (shard 0), not cuda:0 blindly
    dev = model.get_input_embeddings().weight.device
    # left-truncate so the rules + the tail of the comment survive; the yes/no question sits at the end
    tok.truncation_side = "left"
    # first subword of "yes"/"no"; the gap is read at these two vocab ids only
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    rows = split.to_dicts()
    if smoke:
        rows = rows[:12]
    bodies, out, argmax_hits = {}, [], 0
    for ci, r in enumerate(rows):
        s = r["subreddit"]
        # cache each subreddit's comment table once; the split is sorted-ish by subreddit so this stays cheap
        if s not in bodies:
            bodies[s] = K.load_comments(s)
        body = bodies[s][r["idx"]][0]
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        # add_special_tokens=False: _build_prompt already emits the chat template, so don't double-wrap
        enc = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            # next-token logits at the final position; float32 so the yes/no subtraction is stable
            ll = model(**enc, use_cache=False).logits[0, -1, :].to(torch.float32)
        # sanity counter: fraction of prompts where the model's top token is actually yes or no
        argmax_hits += int(int(ll.argmax().item()) in (yes_id, no_id))
        out.append({"subreddit": s, "idx": int(r["idx"]), "label": int(r["label"]),
                    "gap": float((ll[yes_id] - ll[no_id]).item())})
        if smoke or (ci + 1) % 500 == 0:
            print(f"[gap-{tag}] {ci+1}/{len(rows)} argmax_ok={argmax_hits/(ci+1):.3f}", flush=True)
    df = pl.DataFrame(out)
    df.write_parquet(ROOT / f"results/kumar_mod/llm_gap_{tag}.parquet")
    # score per subreddit, then take the median across subreddits (the "balanced" BAL/PR-AUC in Table 1)
    aucs, prs = [], []
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        y = g["label"].to_numpy().astype(int); sc = g["gap"].to_numpy().astype(float)
        # skip degenerate subs: single-class labels, <10 examples, or any non-finite gap would break AUC
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10 or not np.isfinite(sc).all():
            continue
        aucs.append(roc_auc_score(y, sc)); prs.append(average_precision_score(y, sc))
    summ = {"tag": tag, "model": model_id, "n": df.height, "n_subs": len(aucs),
            "argmax_ok": round(argmax_hits / max(len(rows), 1), 4),
            "bal_auc_logitgap_median": round(float(np.median(aucs)), 4) if aucs else None,
            "pr_auc_logitgap_median": round(float(np.median(prs)), 4) if prs else None,
            "note": "FULL test fold (no subsample); yes/no logit gap; device_map=auto multi-card "
                    "(logits identical to the single-card load used for gemma4 / the other 3 families)"}
    json.dump(summ, open(ROOT / f"results/kumar_mod/llm_gap_{tag}.json", "w"), indent=2)
    print(f"[gap] {summ}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.tag, a.model, smoke=a.smoke)


if __name__ == "__main__":
    main()
