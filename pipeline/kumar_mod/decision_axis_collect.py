"""Shared decision-token activation collection on Kumar's balanced corpus -- the ONE GPU collection
that unblocks the three internals lenses (decision-axis invariance, four-way LEACE contrast, naming).

WHAT IS CAPTURED. For each sampled balanced comment we build Kumar's VERBATIM moderation prompt
(kumar_data.messages_for_comment) and append the JSON answer prefix `{"would_moderate": "` so the NEXT
token is the model's yes/no decision (the same trick run_slm_mod inference uses). We then capture, at the
DECISION TOKEN (the last position, which predicts yes/no):
  - the residual-stream vector at layers {12,24,31,41}, float32. The SAE lenses use {24,31,41}
    only; L12 is captured solely for the diff-of-means invariance readout (there is no SAE at L12);
  - the decision gap logit[yes]-logit[no];
  - whether argmax over the vocab at that position is in {yes,no} (LOCUS sanity).
Optionally (--rule_presence) we ALSO forward each comment with the community's specific rules replaced by
a generic "be civil" rule (prompt SHAPE preserved), giving a paired rules-vs-norules capture from which
the rule-presence direction is the difference -- the positive lever for the LEACE contrast (rank 4).

MANDATORY LOCUS SANITY GATE. Before trusting any downstream geometry, on a 200-comment
probe the captured decision-token argmax must be in {yes,no} for >=80% of comments; else the locus is wrong
and we HALT rather than proceed. Run `--gate_only` first; the orchestrator checks the gate JSON before the
full collection.

Storage (results/kumar_mod/decisiontok/):
  meta.parquet           one row per comment: row, subreddit, idx, label, gap_rules, argmax_ok_rules, truncated[, gap_norules, argmax_ok_norules, truncated_norules]
                         (truncated/truncated_norules: prompt exceeded MAX_LEN and was left-truncated -- decision token kept, <bos>/part of prime dropped)
  res_rules_L{L}.fp16.npy   (n, 3840) decision-token residual, real-rules condition (the .fp16 suffix is a
                            fixed filename convention; the stored data is float32)
  res_norules_L{L}.fp16.npy (n, 3840) decision-token residual, generic-rules condition (if --rule_presence;
                            same .fp16 naming convention, data is float32)

  gate:  env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.decision_axis_collect --gate_only --n 200
  full:  env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.decision_axis_collect --per_class 150 --rule_presence
  smoke: env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.decision_axis_collect --smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K


# Capture layers. The SAE lenses use {24,31,41} (capturing at exactly those positions lets the
# same residuals feed the SAE lenses without a re-forward); L12 is captured only for the
# diff-of-means invariance readout, with no SAE at L12. Overridable for cross-model runs
# (DAI_LAYERS=auto rederives the four fractional depths against whatever model is loaded).
LAYERS = [12, 24, 31, 41]
_ENV_LAYERS = os.environ.get("DAI_LAYERS")
if _ENV_LAYERS and _ENV_LAYERS != "auto":
    LAYERS = [int(x) for x in _ENV_LAYERS.split(",")]
OUTDIR = Path(os.environ.get("DAI_OUTDIR", str(ROOT / "results" / "kumar_mod" / "decisiontok")))
GATE_JSON = OUTDIR / "locus_gate.json"
MODEL = os.environ.get("DAI_MODEL", "google/gemma-3-12b-it")
MAX_LEN = 4096
SEED = 11  # the project-wide seed; same value pins the 80/20 split and every bootstrap
# Civility rule that preserves prompt SHAPE while stripping community-specific content;
# the rules-vs-norules residual difference is the rule-presence direction.
GENERIC_RULES = "1. Follow general community guidelines and be respectful."


def _sample(per_class, smoke):
    """Stratified sample: up to per_class removed + per_class kept per subreddit. Returns list of
    (subreddit, idx, body, label) with idx matching K.load_comments order (joins to tox_sent + llm runs)."""
    rng = np.random.default_rng(SEED)
    rows = []
    subs = K.clean_subreddits()
    if smoke:
        subs = subs[:3]
    for s in subs:
        data = K.load_comments(s)
        # i is the position in K.load_comments order, carried through as `idx` so meta.parquet
        # joins back to tox_sent and the LLM-run parquets on (subreddit, idx).
        pos = [(i, b, 1) for i, (b, y) in enumerate(data) if y == 1]
        neg = [(i, b, 0) for i, (b, y) in enumerate(data) if y == 0]
        k = 2 if smoke else per_class
        # Per-class cap drawn separately so removed/kept stay balanced within each subreddit
        # even when the subreddit is itself imbalanced.
        for pool in (pos, neg):
            sel = rng.permutation(len(pool))[:k]
            rows.extend((s, *pool[j]) for j in sel)
    return rows


def _load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto",
                                                 torch_dtype=torch.bfloat16, output_hidden_states=True)
    model.eval()
    return tok, model


def _build_prompt(tok, sub, desc, rules, body):
    msgs = K.messages_for_comment(sub, desc, rules, body)
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Append the JSON answer prefix so the next token to be predicted is the yes/no value
    # itself -- this is the decision token we capture, and it mirrors the inference path.
    return rendered + '{"would_moderate": "'


def _forward_capture(tok, model, prompt, yes_id, no_id, layers):
    """One forward; return (residual_dict{L:(d,)float32 np}, gap, argmax_ok, truncated)."""
    import torch


    # add_special_tokens=False: the chat template already inserted <bos>; re-adding would
    # shift positions. Measure the untruncated length first to record whether truncation fired.
    full_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
    truncated = bool(full_len > MAX_LEN)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN,
              add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    hs = out.hidden_states
    res = {}
    for L in layers:
        # hidden_states[0] is the embedding output, so layer L lives at index L+1.
        idx = L + 1
        if idx >= len(hs):
            raise IndexError(f"layer {L} -> hidden_states[{idx}] out of range (have {len(hs)})")

        # Last position is the decision token; -1 holds the residual that predicts yes/no.
        res[L] = hs[idx][0, -1, :].to(torch.float32).cpu().numpy()
    logits_last = out.logits[0, -1, :].to(torch.float32)
    gap = float((logits_last[yes_id] - logits_last[no_id]).item())
    # Locus check: argmax over the full vocab must land on a yes/no token, confirming the
    # captured position is actually where the model commits to the decision.
    argmax_ok = bool(int(logits_last.argmax().item()) in (yes_id, no_id))
    return res, gap, argmax_ok, truncated


def run(per_class=150, smoke=False, rule_presence=False, gate_only=False, gate_n=200):
    rows = _sample(per_class, smoke)
    if gate_only:
        # Subsample the full stratified set rather than re-sampling, so the gate probe is a
        # representative slice of exactly what the full run would capture.
        rng = np.random.default_rng(SEED)
        rows = [rows[j] for j in rng.permutation(len(rows))[:gate_n]]
    tok, model = _load_model()
    global LAYERS
    nL = int(getattr(model.config, "num_hidden_layers", 48))
    if _ENV_LAYERS == "auto":
        # Cross-model mode: pick layers at fixed relative depths (0.25/0.5/0.65/0.85) so the
        # capture sits at comparable points regardless of the model's layer count.
        LAYERS = sorted({max(1, min(nL, round(f * nL))) for f in (0.25, 0.50, 0.65, 0.85)})
    D = int(getattr(model.config, "hidden_size", 3840))
    # Left-truncate: drop from the prompt's head (<bos>/system prime) so the decision token
    # at the tail always survives.
    tok.truncation_side = "left"
    # First token of "yes"/"no" without a leading space; these ids define the decision gap.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()


    outdir = (OUTDIR.parent / "decisiontok_smoke") if smoke else OUTDIR
    gate_json = outdir / "locus_gate.json"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[collect] {len(rows)} comments, layers {LAYERS}, rule_presence={rule_presence}, "
          f"gate_only={gate_only}; yes={yes_id} no={no_id}", flush=True)

    res_rules = {L: np.zeros((len(rows), D), dtype=np.float32) for L in LAYERS}
    res_norules = {L: np.zeros((len(rows), D), dtype=np.float32) for L in LAYERS} if rule_presence else None
    meta = []
    t0 = time.time()
    for ri, (s, idx, body, label) in enumerate(rows):
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        rr, gap, ok, trunc = _forward_capture(tok, model, p, yes_id, no_id, LAYERS)
        for L in LAYERS:
            res_rules[L][ri] = rr[L]
        m = {"row": ri, "subreddit": s, "idx": idx, "label": int(label),
             "gap_rules": gap, "argmax_ok_rules": ok, "truncated": bool(trunc)}
        if rule_presence and not gate_only:
            # Paired second forward with community rules swapped for the generic civility rule;
            # same comment, prompt shape preserved, so the residual delta isolates rule presence.
            pn = _build_prompt(tok, s, desc[s], GENERIC_RULES, body)
            rn, gapn, okn, truncn = _forward_capture(tok, model, pn, yes_id, no_id, LAYERS)
            for L in LAYERS:
                res_norules[L][ri] = rn[L]
            m["gap_norules"] = gapn; m["argmax_ok_norules"] = okn
            m["truncated_norules"] = bool(truncn)
        meta.append(m)
        if ri and ri % 250 == 0:
            rate = ri / max(1e-9, time.time() - t0)
            print(f"[collect] {ri}/{len(rows)} {rate:.2f}/s eta {(len(rows)-ri)/max(1e-9,rate)/60:.1f}m "
                  f"argmax_ok so far {np.mean([x['argmax_ok_rules'] for x in meta]):.3f}", flush=True)

    # Locus gate: fraction of comments whose decision-token argmax is a yes/no token. Below
    # 0.80 the captured position is not the decision locus and downstream geometry is invalid.
    argmax_ok_frac = float(np.mean([x["argmax_ok_rules"] for x in meta]))
    gate = {"n": len(meta), "argmax_ok_frac": round(argmax_ok_frac, 4), "threshold": 0.80,
            "pass": bool(argmax_ok_frac >= 0.80), "yes_id": int(yes_id), "no_id": int(no_id),
            "mean_gap": round(float(np.mean([x["gap_rules"] for x in meta])), 4)}
    gate_json.parent.mkdir(parents=True, exist_ok=True)
    gate_json.write_text(json.dumps(gate, indent=2))
    print(f"[collect] LOCUS GATE argmax_ok={argmax_ok_frac:.3f} pass={gate['pass']} -> {gate_json}", flush=True)

    if gate_only:
        if not gate["pass"]:
            print("[collect] LOCUS GATE FAILED (<80% argmax in {yes,no}). HALT -- do not run the full "
                  "collection; the decision locus is mis-located.", flush=True)
            sys.exit(2)
        print("[collect] gate passed; safe to run the full collection.", flush=True)
        return

    # Even on a full run the gate is enforced before anything is persisted: a failed locus
    # would silently corrupt every downstream lens, so refuse to write the residuals.
    if not gate["pass"]:
        print("[collect] LOCUS GATE FAILED on the full set (<80% argmax in {yes,no}). HALT -- the decision "
              "locus is mis-located; refusing to write meta.parquet/residuals. locus_gate.json written for "
              "the dashboard.", flush=True)
        sys.exit(2)

    pl.DataFrame(meta).write_parquet(outdir / "meta.parquet")
    for L in LAYERS:
        np.save(outdir / f"res_rules_L{L}.fp16.npy", res_rules[L])
        if rule_presence:
            np.save(outdir / f"res_norules_L{L}.fp16.npy", res_norules[L])
    print(f"[collect] SAVED {len(meta)} rows + residuals -> {outdir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per_class", type=int, default=150, help="removed + kept per subreddit")
    ap.add_argument("--rule_presence", action="store_true", help="also capture generic-rules forward (2x cost)")
    ap.add_argument("--gate_only", action="store_true", help="run only the 200-comment locus sanity gate")
    ap.add_argument("--gate_n", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(per_class=a.per_class, smoke=a.smoke, rule_presence=a.rule_presence,
        gate_only=a.gate_only, gate_n=a.gate_n)


if __name__ == "__main__":
    main()
