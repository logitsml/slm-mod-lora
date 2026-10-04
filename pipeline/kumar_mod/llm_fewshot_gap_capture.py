"""Retrieval-augmented few-shot, scored by the next-token yes/no LOGIT GAP (the static/
random few-shot rung is rating-scored, which under-credits the LLM; the strong test is RETRIEVED exemplars +
gap scoring). For each TEST comment we retrieve its nearest labeled exemplars from the SAME community's TRAIN
fold by e5 cosine similarity (leakage-free: exemplars are train, scored comment is test), demonstrate them
in-context in Kumar's exact schema, then read the yes/no logit gap (parallel to how SLM-Mod and the zero-shot
gap arm are scored). k = k_pos + k_neg balanced (default 2+2), nearest-first, interleaved.

Reuses: decision_axis_collect._load_model (HF, gives logits) + MAX_LEN; kumar_data.{base_chat,comment_turn,
ACK_COMMENT,THIRD_STRING}; fairness_compare.cert_heldout. Model chosen via env DAI_MODEL (as in the zero-shot
gap capture). e5 retrieval uses the cached _fairness_e5_cache.npy (row-aligned to the split).

  CUDA_VISIBLE_DEVICES=N DAI_MODEL=google/gemma-3-12b-it \
     python -m pipeline.kumar_mod.llm_fewshot_gap_capture --tag gemma3_12b [--k 4] [--smoke]
Out: results/kumar_mod/llm_fewshot_gap_{tag}.parquet  and  llm_fewshot_gap_{tag}.json
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
from pipeline.kumar_mod.decision_axis_collect import _load_model, MAX_LEN
from pipeline.kumar_mod.fairness_compare import cert_heldout

SPLIT = ROOT / "results/kumar_mod/balanced/slm_mod_split.parquet"
ECACHE = ROOT / "results/kumar_mod/balanced/_fairness_e5_cache.npy"
TOXPQ = ROOT / "data/processed/kumar_balanced_tox_sent.parquet"
OUTD = ROOT / "results/kumar_mod"
EX_TOK = 96


def _gold_json(label):
    # Demonstration answer for an exemplar, in Kumar's exact output schema. Only would_moderate
    # and rating carry signal; the free-text fields stay empty so the model isn't taught to
    # confabulate rule numbers or explanations. rating 4/1 mirrors the moderate/keep poles.
    return ('{"would_moderate": "%s", "rule": "", "rule_nums": "", "explanation": "", "rating": %d}'
            % ("yes" if label == 1 else "no", 4 if label == 1 else 1))


def _cap(tok, body, budget):
    # Truncate a comment to budget tokens (not chars) so exemplar/target lengths are controlled in
    # the model's own tokenization. Keeps the few-shot block from blowing past MAX_LEN.
    if budget <= 0:
        return ""
    ids = tok(body, add_special_tokens=False)["input_ids"][:budget]
    return tok.decode(ids)


def _fs_chat(s, desc, rules, target_body, exemplars):
    # Each exemplar is a full completed Kumar turn-triple (comment -> ack -> decision request ->
    # gold JSON), so the in-context demonstrations match the live scoring schema exactly.
    chat = list(K.base_chat(s, desc, rules))
    for eb, el in exemplars:
        chat += [{"role": "user", "content": K.comment_turn(eb)},
                 {"role": "assistant", "content": K.ACK_COMMENT},
                 {"role": "user", "content": K.THIRD_STRING},
                 {"role": "assistant", "content": _gold_json(el)}]
    # Target turn is left open after the decision request so the next token is the verdict.
    chat += [{"role": "user", "content": K.comment_turn(target_body)},
             {"role": "assistant", "content": K.ACK_COMMENT},
             {"role": "user", "content": K.THIRD_STRING}]
    return chat


def _build(tok, s, desc, rules, target_body, exemplars):
    rendered = tok.apply_chat_template(_fs_chat(s, desc, rules, target_body, exemplars),
                                       tokenize=False, add_generation_prompt=True)
    # Force the JSON prefix so the very next token must be "yes"/"no" -- that is the position we
    # read the logit gap from, instead of free-generating and parsing.
    return rendered + '{"would_moderate": "'


def run(tag, k=4, smoke=False):
    import torch
    df = pl.read_parquet(SPLIT).select(["subreddit", "idx", "label", "fold", "body"]).with_row_index("row")
    # e5 cache is row-aligned to the split file; the row index ties an embedding back to its comment.
    X = np.load(ECACHE, mmap_mode="r")
    assert X.shape[0] == df.height, f"embed/split mismatch {X.shape[0]} vs {df.height}"
    Xf = np.asarray(X)
    desc, rules = K.load_rules()
    tok, model = _load_model()
    dev = next(model.parameters()).device
    # Keep the right edge (the open verdict turn) when a prompt overflows; drop the oldest context.
    tok.truncation_side = "left"
    # First subword of "yes"/"no"; the gap between these two logits is the moderation score.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    kpos = kneg = max(1, k // 2)

    sub = df["subreddit"].to_numpy(); fold = df["fold"].to_numpy()
    lab = df["label"].to_numpy().astype(int); body = df["body"].to_list()
    comms = sorted(set(sub.tolist()))
    out, argmax_hits, n_done, n_trunc = [], 0, 0, 0
    for s in comms:
        # Retrieve only from this community's TRAIN fold; score only its TEST fold. Same-community
        # retrieval keeps exemplars on-norm; train-vs-test keeps the arm leakage-free.
        tr = np.where((sub == s) & (fold == "train"))[0]
        te = np.where((sub == s) & (fold == "test"))[0]
        pos = tr[lab[tr] == 1]; neg = tr[lab[tr] == 0]
        # Need both classes present to build a balanced demonstration set, and a non-empty test set.
        if len(pos) == 0 or len(neg) == 0 or len(te) == 0:
            continue
        Xpos, Xneg = Xf[pos], Xf[neg]
        if smoke:
            te = te[:6]
        for i in te:
            xi = Xf[i]

            # e5 embeddings are unit-normalized, so the dot product is cosine similarity.
            sp = Xpos @ xi; sn = Xneg @ xi
            # Nearest kpos positives / kneg negatives. The >0.999 guard sends near-duplicates to
            # +inf (sorted last) so an exemplar that is essentially the test comment can't be picked.
            pi = pos[np.argsort(np.where(sp > 0.999, np.inf, -sp))[:kpos]]
            ni = neg[np.argsort(np.where(sn > 0.999, np.inf, -sn))[:kneg]]
            # Interleave pos/neg nearest-first (pos, neg, pos, neg, ...) so neither label clusters at
            # one end of the context; pad with None then drop it to zip the two ranked lists.
            ex_rows = [r for pair in zip(list(pi) + [None] * kneg, list(ni) + [None] * kpos)
                       for r in pair if r is not None]
            exemplars = [(_cap(tok, body[r], EX_TOK), int(lab[r])) for r in ex_rows]
            # Measure the prompt length with an empty target, then give the target whatever token
            # budget is left under MAX_LEN (8-token floor, 8-token slack) so exemplars are never
            # crowded out by a long comment.
            base_len = len(tok(_build(tok, s, desc[s], rules[s], "", exemplars),
                               add_special_tokens=False)["input_ids"])
            tgt = _cap(tok, body[i], max(8, MAX_LEN - base_len - 8))
            prompt = _build(tok, s, desc[s], rules[s], tgt, exemplars)
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            n_trunc += int(len(ids) > MAX_LEN)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
            enc = {kk: v.to(dev) for kk, v in enc.items()}
            with torch.no_grad():
                # Logits over the next token after the forced JSON prefix.
                ll = model(**enc).logits[0, -1, :].to(torch.float32)
            # Sanity track: how often the model's top token is actually yes/no (a degenerate
            # gap is meaningless if the verdict slot wants some other token).
            argmax_hits += int(int(ll.argmax().item()) in (yes_id, no_id))
            out.append({"subreddit": s, "idx": int(df["idx"][int(i)]), "label": int(lab[i]),
                        "gap": float((ll[yes_id] - ll[no_id]).item())})
            n_done += 1
            if smoke or n_done % 500 == 0:
                print(f"[fsgap-{tag}] {n_done} argmax_ok={argmax_hits/n_done:.3f} trunc={n_trunc}", flush=True)

    R = pl.DataFrame(out)
    R.write_parquet(OUTD / f"llm_fewshot_gap_{tag}.parquet")
    # Attach Detoxify scores per comment so the non-toxic AUC can exclude toxic positives.
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    R = R.join(tox, on=["subreddit", "idx"], how="left")
    aucs, prs, ntx, c95, c90, tr95, tr90, ncomm = [], [], [], 0, 0, 0, 0, 0
    # Metrics are computed within community then medianed, never pooled, so big communities don't dominate.
    for s, g in R.group_by("subreddit"):
        y = g["label"].to_numpy().astype(int); sc = g["gap"].to_numpy().astype(float)
        t = g["tox_toxicity"].fill_null(0.5).to_numpy().astype(float)
        # Skip degenerate communities: need both classes, >=10 comments, finite scores.
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10 or not np.isfinite(sc).all():
            continue
        ncomm += 1
        aucs.append(roc_auc_score(y, sc)); prs.append(average_precision_score(y, sc))
        # Non-toxic AUC: keep all negatives but only positives that are NOT toxic (tox < 0.1). Tests
        # whether the gap tracks community norms rather than surface toxicity. Need both classes and
        # at least 3 retained positives for the AUC to mean anything.
        km = (y == 0) | ((y == 1) & (t < 0.1)); yk = y[km]
        if len(np.unique(yk)) == 2 and yk.sum() >= 3:
            ntx.append(roc_auc_score(yk, sc[km]))
        # Certified-coverage at two precision targets; cert_heldout returns (achieved, trivial) where
        # trivial communities are excluded from the denominator below.
        a95, x95 = cert_heldout(y, sc, target=0.95); a90, x90 = cert_heldout(y, sc, target=0.90)
        c95 += int(a95); tr95 += int(x95); c90 += int(a90); tr90 += int(x90)
    med = lambda a: round(float(np.median(a)), 4) if a else None
    summ = {"tag": tag, "model": os.environ.get("DAI_MODEL"), "mode": "fewshot_retrieved_gap",
            "k": k, "k_pos": kpos, "k_neg": kneg, "n": R.height, "n_comm": ncomm,
            "argmax_ok": round(argmax_hits / max(n_done, 1), 4), "n_truncated_over_maxlen": n_trunc,
            "bal_auc_logitgap_median": med(aucs), "pr_auc_logitgap_median": med(prs),
            "nontox_auc_canonical_median": med(ntx),
            # Percent of NON-trivial communities certified at each target (trivial base-rate ones removed).
            "cert@0.95": round(100 * c95 / max(ncomm - tr95, 1), 1),
            "cert@0.90": round(100 * c90 / max(ncomm - tr90, 1), 1),
            "note": ("retrieved few-shot (kNN train exemplars by e5 cosine, balanced, interleaved), scored by "
                     "the yes/no logit gap; leakage-free (exemplars train, scored comments test)")}
    json.dump(summ, open(OUTD / f"llm_fewshot_gap_{tag}.json", "w"), indent=2)
    print(f"[fsgap] {json.dumps(summ, indent=2)}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.tag, k=a.k, smoke=a.smoke)


if __name__ == "__main__":
    main()
