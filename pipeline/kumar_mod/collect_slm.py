"""Collect the FINE-TUNED SLM-Mod model's decision-token activations per community -- tests whether toxicity
collapse appears REPRESENTATIONALLY *inside* the supervised winner. The SLM beats the prompted LLM behaviorally
(BAL-AUC 0.81 vs ~0.67); does it nonetheless encode its keep/remove decision on a generic-toxicity axis, or did
per-community supervision give it genuine community-specific structure? This is the "(a)" leg: re-train 95
per-sub LoRA + collect the fine-tuned 8B's decision token.

METHOD (defensibility). For each community we re-train the per-sub LoRA with run_slm_mod's EXACT recipe
(Llama-3.1-8B base, r16/a32 all-linear LoRA, BOS-stripped prompt/completion, completion-only loss, Appendix-B
hyperparameters). The recipe is COPIED here, NOT imported from / written into run_slm_mod, so the tested
run_slm_mod that the head-to-head depends on is never modified. run_slm_mod uses
save_strategy='no' (adapters are not persisted), so in-process re-training is required -- exactly the substage's
"re-train 95 per-sub LoRA (save adapters) + collect". After training, ONE forward per balanced item captures the
decision-token residual at relative-depth layers via decision_axis_collect._forward_capture VERBATIM (same fp32
precision cast, same left-truncation, same gap/argmax) -> identical res_rules_L{L}.fp16.npy +
meta.parquet schema as the base-model collection, so the invariance / TC_rep analyses consume it unchanged once
pointed at the SLM's layers (recorded in layers.json).

Single-sub-per-process (clean adapter state + crash isolation), mirroring run_slm_mod. A driver loops subs.
Locus sanity gate (decision-token argmax in {yes,no} >= 80%) checked and recorded at aggregate (a sub-threshold
run is flagged in locus_gate.json, not blocked).

  smoke (1 sub, fast):  python -m pipeline.kumar_mod.collect_slm --sub askscience --smoke
  one sub:              python -m pipeline.kumar_mod.collect_slm --sub askscience
  aggregate:            python -m pipeline.kumar_mod.collect_slm --aggregate
Out: results/kumar_mod/decisiontok_slm/{meta.parquet, res_rules_L{L}.fp16.npy, layers.json, locus_gate.json}
     (per-sub shards under decisiontok_slm/_pc/ before aggregate)
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod import run_slm_mod as SLM
from pipeline.kumar_mod.decision_axis_collect import _forward_capture

OUTDIR = ROOT / "results" / "kumar_mod" / "decisiontok_slm"
PC = OUTDIR / "_pc"
SEED = 11  # project-wide seed; same one that fixes the 80/20 split, so item sampling is reproducible across arms
PER_CLASS = 150


# Capture by relative depth, not absolute layer index, so the SLM's layers line up
# with the base model's for the invariance / TC_rep comparison even if depths differ.
REL_DEPTHS = (0.25, 0.50, 0.65, 0.85)


def _layers_for(model):
    nL = int(getattr(model.config, "num_hidden_layers", 32))
    # round each fraction to a layer, clamp into [1, nL], dedup -- two fractions can collapse to one layer
    return sorted({max(1, min(nL, round(f * nL))) for f in REL_DEPTHS})


def _train_adapter(sub, smoke):
    """Re-train sub's LoRA with run_slm_mod's EXACT recipe (copied; run_slm_mod is never modified).
    Returns (tok, model, desc, rules, split) or None if the sub is absent from the split.
    Mirrors run_slm_mod.run_sub training block VERBATIM."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    split_path = SLM.SPLIT_SMOKE if smoke else SLM.SPLIT
    if not split_path.exists():
        SLM.make_split(cap=(60 if smoke else None))
    split = pl.read_parquet(split_path).filter(pl.col("subreddit") == sub)
    if split.height == 0:
        return None
    desc, rules = K.load_rules()
    tok = AutoTokenizer.from_pretrained(SLM.BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right-pad for training; capture later flips to left-truncation
    tr = split.filter(pl.col("fold") == "train")
    bos = tok.bos_token or ""
    prompts_tr, comps = [], []
    for r in tr.iter_rows(named=True):
        msgs = SLM._messages(sub, desc, rules, r["body"])
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Strip the leading BOS the template adds: SFTTrainer re-tokenizes and prepends its own,
        # so leaving it here would double the BOS. Matches run_slm_mod exactly.
        if bos and p.startswith(bos):
            p = p[len(bos):]
        prompts_tr.append(p)
        # Completion-only target is the JSON the model must emit; loss is on these tokens only.
        comps.append('{"would_moderate": "%s"}' % ("yes" if r["label"] == 1 else "no"))
    ds = Dataset.from_dict({"prompt": prompts_tr, "completion": comps})
    model = AutoModelForCausalLM.from_pretrained(SLM.BASE, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    cfg = SFTConfig(output_dir=str(PC / f"_train_{sub}"),
                    num_train_epochs=1, optim="adamw_torch", weight_decay=0.01, warmup_steps=5,
                    lr_scheduler_type="linear", learning_rate=2e-4,
                    # eff. batch 16 via accumulation; per-device=1 keeps the 8B in memory under grad checkpointing
                    per_device_train_batch_size=1, gradient_accumulation_steps=16,
                    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                    # save_strategy='no' mirrors run_slm_mod -- adapters never hit disk, hence in-process re-train
                    logging_steps=50, save_strategy="no", bf16=True, max_length=SLM.MAX_LEN,
                    # keep_end: when a prompt overflows MAX_LEN, drop from the front so the completion survives
                    truncation_mode="keep_end", packing=False, report_to=[])
    print(f"[{sub}] training LoRA on {len(ds)} examples", flush=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()


    # Fold the LoRA into the base weights so capture runs on a plain dense model -- the
    # forward path then matches the base-model collection byte-for-byte (no PEFT hooks in the way).
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.config.output_hidden_states = True  # capture reads residual stream from hidden_states
    model.eval()
    return tok, model, desc, rules, split


def _balanced_items(split, per_class):
    """per_class removed + per_class kept from the sub's split rows (seed 11) -- the items the SLM decides on."""
    rng = np.random.default_rng(SEED)
    pos = [(r["idx"], r["body"]) for r in split.iter_rows(named=True) if r["label"] == 1]
    neg = [(r["idx"], r["body"]) for r in split.iter_rows(named=True) if r["label"] == 0]
    out = []
    for pool, lab in ((pos, 1), (neg, 0)):
        if not pool:
            continue
        # Equal cap per class -> 50/50 keep/remove, so the captured axis isn't confounded by base rate.
        # Take a prefix of a seeded permutation; fewer than per_class available just yields the whole pool.
        sel = rng.permutation(len(pool))[:per_class]
        out.extend((pool[j][0], pool[j][1], lab) for j in sel)
    return out


def collect_sub(sub, smoke=False):
    out_meta = PC / (f"{sub}.smoke.parquet" if smoke else f"{sub}.parquet")
    if not smoke and out_meta.exists():
        print(f"[{sub}] already collected; skip", flush=True); return
    PC.mkdir(parents=True, exist_ok=True)
    import torch
    trained = _train_adapter(sub, smoke)
    if trained is None:
        print(f"[{sub}] not in split; skip", flush=True); return
    tok, model, desc, rules, split = trained
    tok.truncation_side = "left"  # overflow drops <bos>/prompt head; the decision token at the end is preserved
    # First token id of "yes"/"no" -- the two completion values the decision gap is read between.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    layers = _layers_for(model)
    D = int(getattr(model.config, "hidden_size", 4096))


    # Capture on the held-out TEST fold only -- training and probing must not share items.
    items = _balanced_items(split.filter(pl.col("fold") == "test"), per_class=(2 if smoke else PER_CLASS))
    res = {L: np.zeros((len(items), D), dtype=np.float32) for L in layers}
    meta = []
    for ri, (idx, body, label) in enumerate(items):
        msgs = SLM._messages(sub, desc, rules, body)
        # Prime the model up to the open quote so the very next token IS the yes/no decision;
        # the residual at that position is what we capture. Same prime string as training/inference.
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + '{"would_moderate": "'
        rr, gap, ok, trunc = _forward_capture(tok, model, prompt, yes_id, no_id, layers)
        for L in layers:
            res[L][ri] = rr[L]
        meta.append({"row": ri, "subreddit": sub, "idx": int(idx), "label": int(label),
                     "gap_rules": gap, "argmax_ok_rules": ok, "truncated": bool(trunc)})
    suffix = ".smoke" if smoke else ""
    pl.DataFrame(meta).write_parquet(out_meta)
    for L in layers:
        np.save(PC / f"{sub}{suffix}_L{L}.npy", res[L])
    (PC / f"{sub}{suffix}.layers.json").write_text(json.dumps(layers))
    okf = float(np.mean([m["argmax_ok_rules"] for m in meta])) if meta else 0.0
    print(f"[{sub}] collected {len(items)} decision tokens, layers {layers}, argmax_ok {okf:.3f} -> {out_meta}",
          flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()


def aggregate():
    metas = sorted(p for p in PC.glob("*.parquet") if not p.name.endswith(".smoke.parquet"))
    if not metas:
        print("[collect_slm] no per-sub collections to aggregate"); return
    layers = json.loads(sorted(p for p in PC.glob("*.layers.json") if not p.name.endswith(".smoke.layers.json"))[0].read_text())
    big_meta, big_res, row0 = [], {L: [] for L in layers}, 0
    for mp in metas:
        sub = mp.stem
        m = pl.read_parquet(mp)
        # Re-base each sub's per-shard row index onto the global concat offset so meta `row`
        # stays aligned with the stacked activation matrix row-for-row.
        big_meta.append(m.with_columns((pl.col("row") + row0).alias("row")))
        for L in layers:
            big_res[L].append(np.load(PC / f"{sub}_L{L}.npy"))
        row0 += m.height
    meta = pl.concat(big_meta)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta.write_parquet(OUTDIR / "meta.parquet")
    for L in layers:
        # Filename keeps the .fp16 tag for schema parity with the base-model collection, but the
        # array is saved fp32 -- downstream analyses read it as-is, so the dtype here is the truth.
        np.save(OUTDIR / f"res_rules_L{L}.fp16.npy", np.concatenate(big_res[L], axis=0).astype(np.float32))
    okf = float(meta["argmax_ok_rules"].mean())
    (OUTDIR / "layers.json").write_text(json.dumps({"model": SLM.BASE, "layers": layers,
                                                    "hidden_size": int(np.load(PC / f"{metas[0].stem}_L{layers[0]}.npy").shape[1])}))
    # Locus sanity gate: across all captured tokens the vocab argmax must land in {yes,no} >=80%,
    # else we're reading the residual at the wrong position and the decision axis is meaningless.
    gate = {"n": meta.height, "n_subs": len(metas), "argmax_ok_frac": round(okf, 4),
            "threshold": 0.80, "pass": bool(okf >= 0.80)}
    (OUTDIR / "locus_gate.json").write_text(json.dumps(gate, indent=2))
    flag = "" if gate["pass"] else "  !!! LOCUS GATE FAILED (<80% argmax in {yes,no}) -- decision locus suspect"
    print(f"[collect_slm] AGG {meta.height} rows / {len(metas)} subs, layers {layers}, "
          f"argmax_ok {okf:.3f} pass={gate['pass']}{flag} -> {OUTDIR}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sub", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    elif a.sub:
        collect_sub(a.sub, smoke=a.smoke)
    else:
        print("specify --sub S [--smoke] | --aggregate")


if __name__ == "__main__":
    main()
