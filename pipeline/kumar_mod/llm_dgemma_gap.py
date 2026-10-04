"""DiffusionGemma (block-diffusion) yes/no GAP ANALOG -- decoding-paradigm ablation arm.

No exact analog of the AR next-token logit gap exists for a random-init block-diffusion decoder:
the canvas initializes with uniform-random vocab tokens and
the position-0 distribution conditions, through bidirectional attention, on that random draw.
ANALOG: mean over K=4 seeded canvas
initializations of (logit[yes] - logit[no]) read at canvas position 0 from a SINGLE first denoising
step, with the teacher-forced JSON prefix ('{"would_moderate": "') appended to the ENCODER prompt
exactly as in decision_axis_collect._build_prompt. Same fold (slm_mod_split test), same MAX_LEN=4096
left truncation, same yes/no token convention as llm_recency_gap.py -> rows are schema-compatible
with the other llm_gap_* captures and recency_b1.py consumes them unchanged.

  CUDA_VISIBLE_DEVICES=2,3 python -m pipeline.kumar_mod.llm_dgemma_gap \
      --tag dgemma26b --model google/diffusiongemma-26B-A4B-it [--smoke] [--k 4]
Out: results/kumar_mod/llm_gap_{tag}.parquet and .json (recency schema + k / gap_std / analog note).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.decision_axis_collect import _build_prompt, MAX_LEN

# Per-comment canvas seeds are SEED0 + comment_index, kept well clear of the
# project-wide seed-11 split so the random canvas draws can't collide with it.
SEED0 = 11_000_000


def load_dgemma(model_id, eqcheck=False):
    """Materialize on CPU, then place the weight-tied encoder/decoder layers across two GPUs.

    DiffusionGemma ties all 30 decoder layers to the encoder layers (one 25.2B parameter set
    serves both roles), so placement moves each shared layer once and routes activations with
    boundary pre-hooks. Placement only; no computation differs from the reference
    implementation. eqcheck=True compares placed-model logits against a CPU reference forward
    and records the result in the run manifest emitted with every output JSON.
    """
    import os, torch
    from torch import nn
    import transformers, accelerate
    from transformers import DiffusionGemmaForBlockDiffusion
    torch.set_num_threads(max(1, min(32, (os.cpu_count() or 32) // 2)))
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, dtype=torch.bfloat16)
    # Untie the lm_head from the decoder embeddings by replacing it with an
    # independent clone, so moving the head to a separate GPU below can't drag
    # the shared embedding weight off the embedding device.
    model.lm_head.weight = nn.Parameter(model.model.decoder.embed_tokens.weight.detach().clone())
    model.eval()
    tcfg = getattr(model.config, "text_config", model.config)
    vocab = int(tcfg.vocab_size)
    eq = None
    if eqcheck:
        # Reference position-0 logits from the fully-CPU model, captured before
        # any layer is dispatched to GPU; compared against the placed model at
        # the end to confirm placement is bit-faithful (placement only, no math).
        ids = torch.arange(5, 25).unsqueeze(0)
        cv = torch.randint(0, vocab, (1, 256), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            ref = model(input_ids=ids, decoder_input_ids=cv).logits[0, 0, :].float().clone()
    # Split the 30 tied layers down the middle: 0-14 on gpu0, 15-29 on gpu1.
    # Encoder layer i and decoder layer i share one weight set, so they must
    # land on the same device or the tie is silently broken by the move.
    enc_layers = model.model.encoder.language_model.layers
    dec_layers = model.model.decoder.layers
    for i, (el, dl) in enumerate(zip(enc_layers, dec_layers)):
        d = "cuda:0" if i < 15 else "cuda:1"
        el.to(d)
        dl.to(d)
        assert next(dl.parameters()).device == next(el.parameters()).device, f"tie broken layer {i}"
    model.lm_head.to("cuda:1")
    # Sweep up anything still on CPU (embeddings, vision tower, stray norms):
    # final/decoder norms feed the second-half stack on gpu1, everything else
    # to gpu0 alongside the embeddings.
    for name, mod in list(model.named_modules()):
        own = list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False))
        if own and any(t.device.type == "cpu" for t in own):
            tgt = "cuda:1" if (name.endswith(".norm") and ("language_model" in name or "decoder" in name)) else "cuda:0"
            mod.to(tgt)

    def _mv(o, d):
        if torch.is_tensor(o):
            return o.to(d, non_blocking=True)
        if isinstance(o, dict):
            return {kk: _mv(v, d) for kk, v in o.items()}
        if isinstance(o, tuple):
            return tuple(_mv(v, d) for v in o)
        if isinstance(o, list):
            return [_mv(v, d) for v in o]
        return o

    # Pre-hook on each placed module pulls its incoming args/kwargs onto the
    # module's own device, so activations cross the gpu0->gpu1 boundary
    # automatically without an explicit device_map dispatcher.
    def _attach(mod):
        try:
            d = next(mod.parameters()).device
        except StopIteration:
            return
        mod.register_forward_pre_hook(lambda m, a, kw, _d=d: (_mv(a, _d), _mv(kw, _d)), with_kwargs=True)

    for grp in (enc_layers, dec_layers):
        for lay in grp:
            _attach(lay)
    for name, mod in model.named_modules():
        if name.endswith(".norm") or name == "lm_head":
            _attach(mod)
    # lm_head runs on gpu1; bring its logits back to gpu0 so the rest of run()
    # reads everything from a single device.
    model.lm_head.register_forward_hook(lambda m, a, o: o.to("cuda:0"))
    if eqcheck:
        with torch.no_grad():
            gpu = model(input_ids=ids.to("cuda:0"), decoder_input_ids=cv.to("cuda:0")
                        ).logits[0, 0, :].float().cpu()
        eq = {"pos0_logit_max_abs_diff_cpu_vs_dispatched": float((ref - gpu).abs().max()),
              "pos0_logit_mean_abs_diff": float((ref - gpu).abs().mean()),
              "pos0_argmax_agrees": bool(int(ref.argmax()) == int(gpu.argmax()))}
    runtime = {"transformers": transformers.__version__, "accelerate": accelerate.__version__,
               "torch": torch.__version__,
               "load_path": "cpu-materialize + manual duplication-free placement of the shared "
                            "encoder/decoder layers (tied storage moved once) with explicit boundary "
                            "pre-hooks; workaround for transformers#46566 and an accelerate "
                            "dispatch_model tied-twin duplication OOM; placement-only",
               "lm_head": "untied by clone of model.decoder.embed_tokens",
               "device_map": "enc/dec layers 0-14 + embeddings + vision -> gpu0; "
                             "enc/dec layers 15-29 + lm_head -> gpu1",
               "eqcheck": eq}
    return model, vocab, runtime


def run(tag, model_id, k=4, smoke=False):
    import torch
    from transformers import AutoTokenizer
    # Same seed-11 80/20 split as every other arm; score only the held-out test
    # fold so these rows line up with the AR llm_gap_* captures.
    split = pl.read_parquet(ROOT / "results/kumar_mod/balanced/slm_mod_split.parquet").filter(
        pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    tok = AutoTokenizer.from_pretrained(model_id)
    # Left-truncate so the JSON answer prefix at the tail of the prompt always
    # survives MAX_LEN clipping; dropping the head of an over-long prompt is
    # preferred to losing the decision-token context.
    tok.truncation_side = "left"
    model, vocab, runtime = load_dgemma(model_id, eqcheck=smoke)
    print(f"[dgap-{tag}] runtime={json.dumps(runtime)}", flush=True)
    canvas_len = int(getattr(model.config, "canvas_length",
                             getattr(getattr(model.config, "text_config", model.config), "canvas_length", 256)))
    dev = torch.device("cuda:0")
    # Single-token yes/no ids, same convention as llm_recency_gap.py so the gap
    # logit[yes]-logit[no] is comparable across captures.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    print(f"[dgap-{tag}] loaded; vocab={vocab} canvas={canvas_len} yes={yes_id} no={no_id}", flush=True)
    desc, rules = K.load_rules()
    rows = split.to_dicts()
    if smoke:
        rows = rows[:12]
    bodies, out, argmax_hits = {}, [], 0
    t0 = time.time()
    for ci, r in enumerate(rows):
        s = r["subreddit"]
        if s not in bodies:
            bodies[s] = K.load_comments(s)
        body = bodies[s][r["idx"]][0]
        # Encoder prompt carries the teacher-forced JSON prefix; same builder as
        # the decision-axis collector so the conditioning context is identical.
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        enc = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        # One shared prompt, k copies in the batch -- only the random canvas
        # differs across the k rows, so the gap variance is purely the
        # canvas-init noise the analog is meant to average over.
        ids = enc["input_ids"].repeat(k, 1).to(dev)
        am = enc["attention_mask"].repeat(k, 1).to(dev)
        g = torch.Generator().manual_seed(SEED0 + ci)
        # Uniform-random vocab canvas: there is no AR next-token state, so the
        # decoder is seeded with k independent random initializations.
        canvas = torch.randint(0, vocab, (k, canvas_len), generator=g).to(dev)
        with torch.no_grad():
            # Single first denoising step; logits read at canvas position 0,
            # which is where the yes/no answer would be emitted.
            o = model(input_ids=ids, attention_mask=am, decoder_input_ids=canvas)
        ll = o.logits[:, 0, :].to(torch.float32)                  # (k, vocab) at canvas position 0
        gaps = (ll[:, yes_id] - ll[:, no_id]).cpu().numpy()
        amax = ll.argmax(-1).tolist()
        # Sanity diagnostic: count canvas inits whose top token is yes or no,
        # then flag the comment only if a majority of the k inits land on-axis.
        hit = sum(int(a) in (yes_id, no_id) for a in amax)
        argmax_hits += int(hit > k // 2)
        # Per-comment score is the mean gap over the k canvas inits; gap_std
        # records the across-init spread for the analog note (kept only in the
        # _withstd parquet, dropped from the recency-schema output below).
        out.append({"subreddit": s, "idx": int(r["idx"]), "label": int(r["label"]),
                    "gap": float(np.mean(gaps)), "gap_std": float(np.std(gaps))})
        if smoke:
            top = tok.convert_ids_to_tokens([int(a) for a in amax])
            print(f"[dgap-{tag}] {s}/{r['idx']} label={r['label']} gap={np.mean(gaps):+.3f} "
                  f"(std {np.std(gaps):.3f}) argmax_hit={hit}/{k} top={top}", flush=True)
        elif (ci + 1) % 250 == 0:
            sec = (time.time() - t0) / (ci + 1)
            print(f"[dgap-{tag}] {ci+1}/{len(rows)} argmax_ok={argmax_hits/(ci+1):.3f} "
                  f"{sec:.2f}s/comment eta={(len(rows)-ci-1)*sec/3600:.1f}h", flush=True)
    # Canonical output drops gap_std to stay byte-schema-compatible with the
    # AR llm_gap_* parquets that recency_b1.py consumes; the _withstd copy keeps
    # the spread for the analog provenance.
    df = pl.DataFrame([{kk: v for kk, v in r.items() if kk != "gap_std"} for r in out])
    full = pl.DataFrame(out)
    df.write_parquet(ROOT / f"results/kumar_mod/llm_gap_{tag}.parquet")
    full.write_parquet(ROOT / f"results/kumar_mod/llm_gap_{tag}_withstd.parquet")
    # Within-community AUCs (the paper reports their median, not a pooled AUC).
    aucs, prs = [], []
    for s in df["subreddit"].unique().to_list():
        gdf = df.filter(pl.col("subreddit") == s)
        y = gdf["label"].to_numpy().astype(int)
        sc = gdf["gap"].to_numpy().astype(float)
        # Skip communities that can't yield a valid AUC: not both classes
        # present, fewer than 10 comments, or any non-finite gap.
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10 or not np.isfinite(sc).all():
            continue
        aucs.append(roc_auc_score(y, sc))
        prs.append(average_precision_score(y, sc))
    summ = {"tag": tag, "model": model_id, "n": df.height, "n_subs": len(aucs), "k_canvas_inits": k,
            "argmax_ok": round(argmax_hits / max(len(rows), 1), 4),
            "sec_per_comment": round((time.time() - t0) / max(len(rows), 1), 3),
            "bal_auc_logitgap_median": round(float(np.median(aucs)), 4) if aucs else None,
            "pr_auc_logitgap_median": round(float(np.median(prs)), 4) if prs else None,
            "note": "FULL test fold; diffusion gap analog: mean over K seeded uniform-random "
                    "canvas inits of logit[yes]-logit[no] at canvas position 0, single first "
                    "denoising step, teacher-forced JSON prefix in the encoder prompt",
            "runtime": runtime}
    json.dump(summ, open(ROOT / f"results/kumar_mod/llm_gap_{tag}.json", "w"), indent=2)
    print(f"[dgap] {summ}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.tag, a.model, k=a.k, smoke=a.smoke)


if __name__ == "__main__":
    main()
