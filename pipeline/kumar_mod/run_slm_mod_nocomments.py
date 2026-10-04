from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K

OUT = ROOT / "results" / "kumar_mod" / "balanced"
PC = OUT / "slm_mod_pc"; PC.mkdir(parents=True, exist_ok=True)
SPLIT = OUT / "slm_mod_split.parquet"
SPLIT_SMOKE = SPLIT.with_suffix(".smoke.parquet")
BASE = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 11
MAX_LEN = 4096


def make_split(cap=None):
    rng = np.random.default_rng(SEED)
    rows = []
    for s in K.clean_subreddits():
        data = K.load_comments(s)
        if cap:
            pos = [(i, b, y) for i, (b, y) in enumerate(data) if y == 1][:cap // 2]
            neg = [(i, b, y) for i, (b, y) in enumerate(data) if y == 0][:cap // 2]
            items = pos + neg
        else:
            items = [(i, b, y) for i, (b, y) in enumerate(data)]
        for lab in (0, 1):
            grp = [(i, b) for i, b, y in items if y == lab]
            order = rng.permutation(len(grp))
            ntest = max(1, int(round(0.2 * len(grp))))
            test_local = set(order[:ntest].tolist())
            for k, (idx, b) in enumerate(grp):
                rows.append({"subreddit": s, "idx": idx, "body": b, "label": lab,
                             "fold": "test" if k in test_local else "train"})
    df = pl.DataFrame(rows)

    out = SPLIT if cap is None else SPLIT_SMOKE
    df.write_parquet(out)
    print(f"[split] {df.height} rows, {df['subreddit'].n_unique()} subs, "
          f"test frac {df.filter(pl.col('fold')=='test').height/df.height:.2f} -> {out}", flush=True)
    return df


def _messages(s, desc, rules, body):
    return K.messages_for_comment(s, desc[s], rules[s], body)


def run_sub(sub, smoke=False):
    if not smoke and (PC / f"{sub}.parquet").exists():
        print(f"[{sub}] already done; skip", flush=True); return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    split_path = SPLIT_SMOKE if smoke else SPLIT
    if not split_path.exists():
        make_split(cap=(60 if smoke else None))
    split = pl.read_parquet(split_path).filter(pl.col("subreddit") == sub)
    if split.height == 0:
        print(f"[{sub}] not in split; skip", flush=True); return
    desc, rules = K.load_rules()
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    tr = split.filter(pl.col("fold") == "train")

    bos = tok.bos_token or ""
    prompts_tr, comps = [], []
    for r in tr.iter_rows(named=True):
        msgs = _messages(sub, desc, rules, r["body"])
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if bos and p.startswith(bos):
            p = p[len(bos):]
        prompts_tr.append(p)
        comps.append('{"would_moderate": "%s"}' % ("yes" if r["label"] == 1 else "no"))
    ds = Dataset.from_dict({"prompt": prompts_tr, "completion": comps})
    if smoke:
        bid = tok.bos_token_id
        tr_ids = tok(prompts_tr[0], add_special_tokens=True).input_ids
        inf0 = (tok.apply_chat_template(_messages(sub, desc, rules, "x"), tokenize=False,
                                        add_generation_prompt=True) + '{"would_moderate": "')
        inf_ids = tok(inf0, add_special_tokens=False).input_ids
        assert tr_ids[:2] != [bid, bid] and tr_ids.count(bid) == inf_ids.count(bid) == 1, \
            f"BOS skew: train={tr_ids.count(bid)} inf={inf_ids.count(bid)} first3={tr_ids[:3]}"
        print(f"[{sub}] BOS-parity OK (train==inf==1)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    cfg = SFTConfig(output_dir=str(PC / f"_train_{sub}"),
                    num_train_epochs=1,
                    optim="adamw_torch", weight_decay=0.01, warmup_steps=5, lr_scheduler_type="linear",
                    learning_rate=2e-4,

                    per_device_train_batch_size=1, gradient_accumulation_steps=16,
                    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                    logging_steps=50, save_strategy="no", bf16=True, max_length=MAX_LEN,
                    truncation_mode="keep_end",

                    packing=False, report_to=[])
    print(f"[{sub}] training LoRA on {len(ds)} examples", flush=True)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()

    import gc
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    te = split.filter(pl.col("fold") == "test")
    model.eval()
    tok.padding_side = "left"; tok.truncation_side = "left"
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    rec = []; rows = list(te.iter_rows(named=True)); bs = 2
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        prompts = [tok.apply_chat_template(_messages(sub, desc, rules, r["body"]),
                                           tokenize=False, add_generation_prompt=True) + '{"would_moderate": "'
                   for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        gap = (logits[:, yes_id] - logits[:, no_id]).float().cpu().numpy()
        for r, gp in zip(chunk, gap):
            rec.append({"subreddit": sub, "idx": r["idx"], "label": r["label"],
                        "would_moderate": int(gp > 0), "gap": float(gp)})

    pc_out = PC / (f"{sub}.smoke.parquet" if smoke else f"{sub}.parquet")
    pl.DataFrame(rec).write_parquet(pc_out)
    from sklearn.metrics import balanced_accuracy_score
    y = np.array([r["label"] for r in rec]); p = np.array([r["would_moderate"] for r in rec])
    bacc = balanced_accuracy_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    print(f"[{sub}] {len(rec)} test preds, balanced acc {bacc:.3f} -> {pc_out}", flush=True)


def aggregate():

    parts = [pl.read_parquet(p) for p in sorted(PC.glob("*.parquet")) if not p.name.endswith(".smoke.parquet")]
    if not parts:
        print("no per-sub SLM-Mod parquets"); return
    df = pl.concat(parts)
    df.write_parquet(OUT / "slm_mod_test.parquet")
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    baccs, aucs = [], []
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        y = g["label"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        baccs.append(balanced_accuracy_score(y, g["would_moderate"].to_numpy()))
        aucs.append(roc_auc_score(y, g["gap"].to_numpy()))
    print(f"[slm-mod] {df.height} preds over {df['subreddit'].n_unique()} subs | "
          f"median per-sub balanced-acc {np.median(baccs):.4f} | median per-sub AUC(gap) {np.median(aucs):.4f}"
          f" -> {OUT / 'slm_mod_test.parquet'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--make_split", action="store_true")
    ap.add_argument("--sub", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.make_split:
        make_split(cap=(60 if a.smoke else None))
    elif a.sub:
        run_sub(a.sub, smoke=a.smoke)
    elif a.aggregate:
        aggregate()
    else:
        print("specify --make_split | --sub S | --aggregate")


if __name__ == "__main__":
    main()
