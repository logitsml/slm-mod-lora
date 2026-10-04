"""Full-corpus yes/no LOGIT-GAP capture for gemma-4-12B (the recency arm's gemma-4 only had a per-community
HF-capped subsample; this scores the SAME full 80/20 test fold as the gemma-3/llama/qwen gap captures, so it
is directly comparable). Needs transformers 5.x for gemma-4. Lean single-card load
(no device_map -> no accelerate dependency; hidden_states off). The gap = logit[yes]-logit[no] is identical to
how the other three were captured (load placement / hidden_states do not change logits).

  CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.llm_gemma4_gap \
     --tag gemma4_12b --model google/gemma-4-12B-it [--smoke]
Out: results/kumar_mod/llm_gap_{tag}.parquet  and  llm_gap_{tag}.json   (same names/schema as the other 3)
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
    # Same seed-11 80/20 split as every other arm; score only the held-out test fold so the gap is comparable.
    split = pl.read_parquet(ROOT / "results/kumar_mod/balanced/slm_mod_split.parquet").filter(
        pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    tok = AutoTokenizer.from_pretrained(model_id)

    # Single-card load, no device_map: avoids the accelerate dependency. Logits are unaffected by placement.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    model.eval()
    dev = next(model.parameters()).device
    # Left-truncate so the prompt tail (the comment + answer cue) survives the length cap.
    tok.truncation_side = "left"
    # First subword of "yes"/"no"; the gap is read off these two vocab logits at the final position.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    rows = split.to_dicts()
    if smoke:
        rows = rows[:12]
    # argmax_hits tracks how often the model's top token is actually yes/no -- a sanity check that the
    # gap is reading a real decision and not noise from an off-format generation.
    bodies, out, argmax_hits = {}, [], 0
    for ci, r in enumerate(rows):
        s = r["subreddit"]
        # Load each community's comments once; rows arrive grouped enough that this cache stays small.
        if s not in bodies:
            bodies[s] = K.load_comments(s)
        body = bodies[s][r["idx"]][0]
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        enc = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            # Logits at the last position in fp32; the next token is the model's yes/no answer.
            ll = model(**enc, use_cache=False).logits[0, -1, :].to(torch.float32)
        argmax_hits += int(int(ll.argmax().item()) in (yes_id, no_id))
        # Signed logit gap = the continuous decision axis; sign and magnitude both matter for AUC.
        out.append({"subreddit": s, "idx": int(r["idx"]), "label": int(r["label"]),
                    "gap": float((ll[yes_id] - ll[no_id]).item())})
        if smoke or (ci + 1) % 500 == 0:
            print(f"[gap-{tag}] {ci+1}/{len(rows)} argmax_ok={argmax_hits/(ci+1):.3f}", flush=True)
    df = pl.DataFrame(out)
    df.write_parquet(ROOT / f"results/kumar_mod/llm_gap_{tag}.parquet")
    # Per-community AUC, then take the median across communities (the paper's within-community metric).
    aucs, prs = [], []
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        y = g["label"].to_numpy().astype(int); sc = g["gap"].to_numpy().astype(float)
        # Skip communities that lack both classes, are too small to be stable, or have any non-finite gap.
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10 or not np.isfinite(sc).all():
            continue
        aucs.append(roc_auc_score(y, sc)); prs.append(average_precision_score(y, sc))
    summ = {"tag": tag, "model": model_id, "n": df.height, "n_subs": len(aucs),
            "argmax_ok": round(argmax_hits / max(len(rows), 1), 4),
            "bal_auc_logitgap_median": round(float(np.median(aucs)), 4) if aucs else None,
            "pr_auc_logitgap_median": round(float(np.median(prs)), 4) if prs else None,
            "note": "FULL test fold (no subsample); yes/no logit gap; lean single-card load (logits identical to "
                    "the device_map load used for the other 3 families)"}
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
