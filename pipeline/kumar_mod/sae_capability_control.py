"""SAE CAPABILITY-PRESERVATION CONTROL -- does ablating the pre-registered toxicity SAE features leave
gemma-3-12b-it's GENERAL competence intact? This is the specificity companion to the necessity test
(sae_causal_dosedense.py): the causal-toxicity claim is only defensible if the SAME live-forward-pass
ablation that flips moderation decisions does NOT (a) inflate language-model perplexity or (b) degrade
unrelated factual yes/no accuracy. If TOX ablation broke the LM broadly, the moderation flips would be a
model-breaking artifact, not evidence of a toxicity-specific decision circuit.

Regenerates results/kumar_mod/sae_capability_control.json, which backs the capability-preservation
control (TOX ablation leaves general perplexity and unrelated factual accuracy intact).

Reuses the EXACT machinery of the necessity test: same _load_model + MAX_LEN truncation mechanics
(decision_axis_collect; _build_prompt is not needed -- the probe builds its own chat-template prompts),
same _load_sae, same JumpReLU encode, same hook subtracting alpha*(acts[F] @ w_dec[F]) at L24+L31+L41 with
the fp32-cast-subtract-cast-back numerics (copied byte-for-byte from sae_causal_dosedense.mk). Frozen,
pre-registered feature sets come from results/kumar_mod/sae/featsets_L{L}_w16k.json -- never
reselected here.

TWO PROBES:
  (a) PERPLEXITY on the frozen WikiText-1k holdout. Arms: clean, TOX (all-position, alpha=1.0), DECMATCH
      (decision-matched non-tox), COMM (Waller-orthogonal community), RANDOM (rotating d = idx % 20).
      Token-weighted corpus PPL (perplexity_from_nlls below) + pct_change vs clean.
      The holdout is identified by its observed sha256, recorded in the output for replication.
  (b) FACTUAL YES/NO capability probe (capability_probe_yesno.json, 200 balanced items, zero toxicity or
      moderation content). Arms: base, TOX all-position, TOX decision-token-only, DECMATCH, COMM,
      RANDOM x3 (rotating), plus a free-form scaffold arm (answers "the result is a JSON-scaffold
      artifact"). accuracy by sign(gap=logit[yes]-logit[no]), flip vs base, mean |delta-gap|;
      question-resampling bootstrap CI95 on acc_base-acc_TOX and flip_TOX-flip_RANDOM.

PRE-REGISTERED ACCEPTANCE CRITERIA are written INTO the output JSON *before* the metrics are computed
(the acceptance_criteria block). These SCIENTIFIC criteria MAY legitimately fail -- a real degradation is
a real result -- so their GATE lines print PASS/FAIL but do NOT change the exit code. MECHANICAL gates
(locus, shapes, sha-recorded, refuse-if-output-exists) DO exit nonzero on failure.

  smoke: python -m pipeline.kumar_mod.sae_capability_control --smoke      # 20 paragraphs, 20 questions -> *_smoke.json
  run:   CUDA_VISIBLE_DEVICES=N python -m pipeline.kumar_mod.sae_capability_control
Out: results/kumar_mod/sae_capability_control.json  (NEW; refuses to overwrite)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod.decision_axis_collect import _load_model, MAX_LEN               # noqa: E402

FSDIR = ROOT / "results" / "kumar_mod" / "sae"                # frozen pre-registered feature sets
WIKIPQ = ROOT / "data" / "processed" / "wikitext103_test_1k.parquet"
OUT = ROOT / "results" / "kumar_mod" / "sae_capability_control.json"
REPO = "google/gemma-scope-2-12b-it"
LAYERS = [24, 31, 41]
WIDTH = "16k"
SEED = 11
BOOT_SEED = 12            # probe question-resampling bootstrap seed (per plan)
N_BOOT = 2000
N_RAND_SETS = 20         # frozen RANDOM draws in the featset file
R_RAND_PROBE = 3         # rotating random arms scored on the probe
PROBE_INSTRUCTION = ('Answer the following question. Return a JSON object with one field "answer" that '
                     'is either "yes" or "no". Question: ')
PROBE_PRIME = '{"answer": "'
FREEFORM_PREFIX = "Answer yes or no: "
FREEFORM_PRIME = "Answer:"


def perplexity_from_nlls(nlls_per_paragraph):
    """Aggregate per-paragraph token-level NLLs into corpus perplexity.

    Definition: corpus PPL = exp(total_NLL / total_tokens). This is the standard token-weighted
    corpus perplexity (NOT the geometric mean of per-paragraph perplexities). Kept inline so this
    module is self-contained in the artifact.
    """
    total_nll = 0.0
    total_tok = 0
    for nlls in nlls_per_paragraph:
        nlls = np.asarray(nlls).ravel()
        total_nll += float(nlls.sum())
        total_tok += len(nlls)
    if total_tok == 0:
        return float("nan")
    return float(np.exp(total_nll / total_tok))


def _load_sae(L, dev):
    # byte-for-byte from sae_causal_dosedense._load_sae (gemma-scope-2 JumpReLU resid_post params, fp32)
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{WIDTH}_l0_medium/params.safetensors")
    sd = load_file(p)
    return {k: sd[k].to(dev, torch.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_probe() -> Path:
    """Locate the authored 200-item probe. Env override first, then the copy next to this module,
    then the shipped copy under data/processed/."""
    cands = []
    if os.environ.get("MMM_PROBE"):
        cands.append(Path(os.environ["MMM_PROBE"]))
    here = Path(__file__).resolve().parent
    cands += [here / "capability_probe_yesno.json", ROOT / "data" / "processed" / "capability_probe_yesno.json"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"capability_probe_yesno.json not found in any of: {[str(c) for c in cands]}")


def _boot_diff(a, b, seed, nboot):
    """Question-resampling bootstrap on mean(a)-mean(b) (paired per-question arrays). CI95."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n = len(a); rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    draws = np.empty(nboot, dtype=np.float64)
    for i in range(nboot):
        ix = rng.integers(0, n, n)
        draws[i] = a[ix].mean() - b[ix].mean()
    return {"point": round(point, 4),
            "ci95": [round(float(np.percentile(draws, 2.5)), 4),
                     round(float(np.percentile(draws, 97.5)), 4)]}


def run(smoke=False):
    import torch

    out_path = OUT.with_name(OUT.stem + "_smoke.json") if smoke else OUT
    # NOTHING may overwrite an existing output -- refuse (mechanical gate; exit nonzero).
    if out_path.exists():
        print(f"GATE refuse_output_exists FAIL {out_path} already exists -- refusing to overwrite", flush=True)
        sys.exit(1)
    print(f"GATE refuse_output_exists PASS {out_path} does not pre-exist", flush=True)

    probe_path = _resolve_probe()
    probe_sha = _sha256_file(probe_path)
    probe = json.loads(probe_path.read_text())
    questions = probe["questions"]
    if smoke:
        # first 10 yes + first 10 no IN FILE ORDER: the probe file groups yes-blocks first per
        # category, so a plain [:20] slice would be all-yes and the smoke accuracy path degenerate.
        questions = ([q for q in questions if q["answer"] == "yes"][:10]
                     + [q for q in questions if q["answer"] == "no"][:10])
    q_text = [q["question"] for q in questions]
    y_true = np.array([1 if q["answer"] == "yes" else 0 for q in questions], dtype=np.int64)  # 1=yes

    # mechanical balance gate: probe must match the plan's pre-verified counts (200 = 100 yes +
    # 100 no; smoke slice 20 = 10 + 10) -- checked BEFORE any GPU work is spent on a wrong file.
    exp_n, exp_yes = (20, 10) if smoke else (200, 100)
    bal_yes = int((y_true == 1).sum()); bal_no = int((y_true == 0).sum())
    balance_ok = (len(questions) == exp_n and bal_yes == exp_yes and bal_no == exp_n - exp_yes)
    if not balance_ok:
        print(f"GATE probe_balance FAIL n={len(questions)} yes={bal_yes} no={bal_no} "
              f"expected n={exp_n} yes={exp_yes} no={exp_n - exp_yes}", flush=True)
        sys.exit(1)

    wiki_sha = _sha256_file(WIKIPQ)
    paras = pl.read_parquet(WIKIPQ)["text"].to_list()
    if smoke:
        paras = paras[:20]

    tok, model = _load_model()
    dev = next(model.parameters()).device
    layers_mod = (model.model.language_model.layers if hasattr(model.model, "language_model")
                  else model.model.layers)
    tok.truncation_side = "left"
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    SAE = {L: _load_sae(L, dev) for L in LAYERS}
    FS = {L: json.loads((FSDIR / f"featsets_L{L}_w{WIDTH}.json").read_text()) for L in LAYERS}

    def T(idx_list):
        return torch.tensor(idx_list, device=dev, dtype=torch.long)

    SET = {L: {"tox": T(FS[L]["TOX"]), "decmatch": T(FS[L]["DECISION_MATCHED_NONTOX"]),
               "comm": T(FS[L]["CLEAN_COMMUNITY"]),
               "rand": [T(r) for r in FS[L]["RANDOM"]]} for L in LAYERS}

    def encode(x, L):
        # JumpReLU: pre-activation gated by the learned per-feature threshold (byte-for-byte w/ siblings)
        s = SAE[L]; pre = x @ s["w_enc"] + s["b_enc"]; return pre * (pre > s["threshold"])

    def mk(L, F, alpha, dectok, counter=None):
        # hook body copied byte-for-byte from sae_causal_dosedense.mk (fp32-cast-subtract-cast-back);
        # `counter` is an added measurement-only side-effect (TOX fire rate) that does NOT touch the arithmetic.
        wdec = SAE[L]["w_dec"]
        def hk(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.to(torch.float32)
            a = encode(hf, L)
            contrib = a[..., F] @ wdec[F]
            if counter is not None:
                counter["fire"] += float((a[..., F] > 0).sum().item())
                counter["tok"] += int(a.shape[0] * a.shape[1])
            if dectok:
                c2 = torch.zeros_like(contrib); c2[:, -1, :] = contrib[:, -1, :]; contrib = c2
            hf = hf - alpha * contrib
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return hk

    # ---- probe forward (last-position logit gap = logit[yes]-logit[no]) --------------------------
    def probe_gap(prompt, hooks):
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN,
                  add_special_tokens=False)   # chat template already supplies BOS
        enc = {k: v.to(dev) for k, v in enc.items()}
        handles = [layers_mod[L].register_forward_hook(hk) for L, hk in hooks]
        try:
            with torch.no_grad():
                out = model(**enc, use_cache=False, output_hidden_states=False)
        finally:
            for hd in handles:
                hd.remove()
        ll = out.logits[0, -1, :].to(torch.float32)
        return float((ll[yes_id] - ll[no_id]).item()), int(ll.argmax().item())

    def probe_prompt(q):
        rendered = tok.apply_chat_template([{"role": "user", "content": PROBE_INSTRUCTION + q}],
                                           tokenize=False, add_generation_prompt=True)
        return rendered + PROBE_PRIME

    def freeform_prompt(q):
        rendered = tok.apply_chat_template([{"role": "user", "content": FREEFORM_PREFIX + q + "."}],
                                           tokenize=False, add_generation_prompt=True)
        return rendered + FREEFORM_PRIME

    # ---- (b.0) probe BASE arm + LOCUS GATE (run FIRST so a mis-located locus aborts before the
    #            expensive PPL/ablation sweeps) -----------------------------------------------------
    base_gap = np.zeros(len(questions)); base_argmax = np.zeros(len(questions), dtype=np.int64)
    for qi, q in enumerate(q_text):
        g, am = probe_gap(probe_prompt(q), [])
        base_gap[qi] = g; base_argmax[qi] = am
    locus_frac = float(np.mean((base_argmax == yes_id) | (base_argmax == no_id)))
    locus_pass = bool(locus_frac >= 0.95)
    if not locus_pass:
        print(f"GATE locus FAIL base_argmax_in_yesno={locus_frac:.4f} < 0.95 -- decision locus mis-located; "
              f"aborting before ablation sweeps", flush=True)
        sys.exit(1)

    # ---- (a) PERPLEXITY arms on the frozen WikiText holdout --------------------------------------
    tox_counter = {L: {"fire": 0.0, "tok": 0} for L in LAYERS}

    def ppl_hooks(arm, pidx):
        if arm == "clean":
            return []
        if arm == "tox":
            return [(L, mk(L, SET[L]["tox"], 1.0, False, tox_counter[L])) for L in LAYERS]
        if arm == "decmatch":
            return [(L, mk(L, SET[L]["decmatch"], 1.0, False)) for L in LAYERS]
        if arm == "comm":
            return [(L, mk(L, SET[L]["comm"], 1.0, False)) for L in LAYERS]
        if arm == "random":
            d = pidx % N_RAND_SETS
            return [(L, mk(L, SET[L]["rand"][d], 1.0, False)) for L in LAYERS]
        raise ValueError(arm)

    def para_nlls(ids, hooks):
        handles = [layers_mod[L].register_forward_hook(hk) for L, hk in hooks]
        try:
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False, output_hidden_states=False).logits
        finally:
            for hd in handles:
                hd.remove()
        shift_logits = logits[0, :-1, :].to(torch.float32)
        shift_labels = ids[0, 1:]
        nll = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none")
        return nll.detach().to(torch.float32).cpu().numpy()

    ppl_arms = ["clean", "tox", "decmatch", "comm", "random"]
    nll_store = {a: [] for a in ppl_arms}
    n_para_used = 0
    for pidx, text in enumerate(paras):
        ids = tok(text, return_tensors="pt")["input_ids"].to(dev)   # default add_special_tokens (BOS)
        if ids.shape[1] < 2:
            continue                                                # no next-token target -> no NLL
        n_para_used += 1
        for a in ppl_arms:
            nll_store[a].append(para_nlls(ids, ppl_hooks(a, pidx)))
        if smoke or (pidx + 1) % 100 == 0:
            print(f"[ppl] {pidx+1}/{len(paras)} used={n_para_used}", flush=True)

    ppl = {a: perplexity_from_nlls(nll_store[a]) for a in ppl_arms}
    ppl_pct = {a: (float(ppl[a] / ppl["clean"] - 1.0) * 100.0 if ppl["clean"] > 0 else float("nan"))
               for a in ppl_arms}
    total_tok = tox_counter[LAYERS[0]]["tok"] or 1
    tox_fire_per_1k = {f"L{L}": round(tox_counter[L]["fire"] / (tox_counter[L]["tok"] or 1) * 1000.0, 3)
                       for L in LAYERS}
    tox_fire_per_1k["total_all_layers"] = round(
        sum(tox_counter[L]["fire"] for L in LAYERS) / total_tok * 1000.0, 3)

    # ---- (b) probe ABLATION arms (tox / tox_dectok / decmatch / comm / random x3 / free-form) -----
    def probe_hooks(arm):
        if arm == "tox":
            return [(L, mk(L, SET[L]["tox"], 1.0, False)) for L in LAYERS]
        if arm == "tox_dectok":
            return [(L, mk(L, SET[L]["tox"], 1.0, True)) for L in LAYERS]
        if arm == "decmatch":
            return [(L, mk(L, SET[L]["decmatch"], 1.0, False)) for L in LAYERS]
        if arm == "comm":
            return [(L, mk(L, SET[L]["comm"], 1.0, False)) for L in LAYERS]
        raise ValueError(arm)

    gaps = {a: np.zeros(len(questions)) for a in ("tox", "tox_dectok", "decmatch", "comm")}
    rand_gaps = np.zeros((R_RAND_PROBE, len(questions)))
    ff_base = np.zeros(len(questions)); ff_tox = np.zeros(len(questions))
    for qi, q in enumerate(q_text):
        pp = probe_prompt(q)
        for a in ("tox", "tox_dectok", "decmatch", "comm"):
            gaps[a][qi], _ = probe_gap(pp, probe_hooks(a))
        for k in range(R_RAND_PROBE):
            d = (qi + k) % N_RAND_SETS
            rand_gaps[k, qi], _ = probe_gap(pp, [(L, mk(L, SET[L]["rand"][d], 1.0, False)) for L in LAYERS])
        fp = freeform_prompt(q)
        ff_base[qi], _ = probe_gap(fp, [])
        ff_tox[qi], _ = probe_gap(fp, [(L, mk(L, SET[L]["tox"], 1.0, False)) for L in LAYERS])
        if smoke or (qi + 1) % 50 == 0:
            print(f"[probe] {qi+1}/{len(questions)}", flush=True)

    # ---- metrics: accuracy by sign(gap), flip vs base, mean |delta-gap| --------------------------
    def pred(g):
        return (np.asarray(g) > 0).astype(np.int64)     # gap>0 -> predicted "yes"

    def acc(g):
        return float(np.mean(pred(g) == y_true))

    def correct_vec(g):
        return (pred(g) == y_true).astype(np.float64)

    def flip_vec(g):
        return (np.sign(np.asarray(g)) != np.sign(base_gap)).astype(np.float64)

    def flip(g):
        return float(np.mean(flip_vec(g)))

    def mean_abs_dgap(g):
        return float(np.mean(np.abs(np.asarray(g) - base_gap)))

    rand_flip_vec = flip_vec(rand_gaps[0])
    for k in range(1, R_RAND_PROBE):
        rand_flip_vec = rand_flip_vec + flip_vec(rand_gaps[k])
    rand_flip_vec = rand_flip_vec / R_RAND_PROBE
    rand_correct_vec = np.mean([correct_vec(rand_gaps[k]) for k in range(R_RAND_PROBE)], axis=0)
    rand_acc = float(rand_correct_vec.mean())
    rand_flip = float(rand_flip_vec.mean())
    rand_mean_abs_dgap = float(np.mean([mean_abs_dgap(rand_gaps[k]) for k in range(R_RAND_PROBE)]))

    acc_base = acc(base_gap)
    probe_metrics = {
        "base": {"accuracy": round(acc_base, 4)},
        "tox": {"accuracy": round(acc(gaps["tox"]), 4), "flip_vs_base": round(flip(gaps["tox"]), 4),
                "mean_abs_delta_gap": round(mean_abs_dgap(gaps["tox"]), 4)},
        "tox_dectok": {"accuracy": round(acc(gaps["tox_dectok"]), 4),
                       "flip_vs_base": round(flip(gaps["tox_dectok"]), 4),
                       "mean_abs_delta_gap": round(mean_abs_dgap(gaps["tox_dectok"]), 4)},
        "decmatch": {"accuracy": round(acc(gaps["decmatch"]), 4),
                     "flip_vs_base": round(flip(gaps["decmatch"]), 4),
                     "mean_abs_delta_gap": round(mean_abs_dgap(gaps["decmatch"]), 4)},
        "comm": {"accuracy": round(acc(gaps["comm"]), 4), "flip_vs_base": round(flip(gaps["comm"]), 4),
                 "mean_abs_delta_gap": round(mean_abs_dgap(gaps["comm"]), 4)},
        "random_x3": {"accuracy": round(rand_acc, 4), "flip_vs_base": round(rand_flip, 4),
                      "mean_abs_delta_gap": round(rand_mean_abs_dgap, 4)},
        "freeform_base": {"accuracy": round(acc(ff_base), 4)},
        "freeform_tox": {"accuracy": round(acc(ff_tox), 4),
                         "flip_vs_freeform_base": round(
                             float(np.mean(np.sign(ff_tox) != np.sign(ff_base))), 4),
                         "mean_abs_delta_gap_vs_freeform_base": round(
                             float(np.mean(np.abs(ff_tox - ff_base))), 4)},
    }

    nboot = 200 if smoke else N_BOOT
    boot = {
        "acc_base_minus_acc_tox": _boot_diff(correct_vec(base_gap), correct_vec(gaps["tox"]),
                                             BOOT_SEED, nboot),
        # plan pins seed 12 for the question-resampling bootstrap; BOTH CIs use it (fresh rng each)
        "flip_tox_minus_flip_random": _boot_diff(flip_vec(gaps["tox"]), rand_flip_vec,
                                                 BOOT_SEED, nboot),
    }

    # ---- (c) PRE-REGISTERED acceptance criteria (declared BEFORE evaluating them) -----------------
    acceptance_criteria = {
        "ppl_tox_within_pct": 5.0,
        "ppl_materiality_margin_pct": 1.0,
        "ppl_not_worse_than": ["decmatch", "random"],
        "probe_acc_drop_soft_pp": 2.0,
        "probe_acc_drop_hard_pp": 5.0,
        "probe_acc_drop_ci_must_contain_zero": True,
        "probe_flip_max": 0.05,
        "note": ("Declared in-file before metrics. TOX ablation preserves capability iff: PPL pct_change(TOX) "
                 "within +/-5% AND not materially worse (>1pp) than the decmatch/random controls; probe "
                 "accuracy drop (base-tox) <= 2pp with bootstrap CI95 containing 0 (hard ceiling 5pp); probe "
                 "flip rate <= 0.05. These are SCIENTIFIC gates -- they may fail without a nonzero exit."),
    }

    # ---- assemble output --------------------------------------------------------------------------
    interpretation = (
        "Capability is preserved iff the toxicity ablation that flips moderation decisions leaves both "
        "general perplexity (+/-5%, no worse than decmatch/random) and unrelated factual yes/no accuracy "
        "(<=2pp drop, CI containing 0) intact, with probe flip <=0.05. TOX at matched positions vs the "
        "decmatch / community / random controls isolates whether any degradation is toxicity-feature "
        "specific or a generic residual-perturbation effect; the decision-token-only and free-form arms "
        "answer the 'saturation' and 'JSON-scaffold artifact' objections respectively. A genuine capability "
        "hit is reported as a real result, not suppressed."
    )
    result = {
        "analysis": "sae_capability_preservation_control",
        "model": "google/gemma-3-12b-it", "sae": REPO, "width": WIDTH, "layers": LAYERS,
        "seed": SEED, "smoke": bool(smoke),
        "acceptance_criteria": acceptance_criteria,
        "provenance": {
            "featsets_dir": str(FSDIR),
            "K_tox_per_layer": {f"L{L}": len(FS[L]["TOX"]) for L in LAYERS},
            "probe_file": str(probe_path), "probe_sha256": probe_sha,
            "probe_n": len(questions), "probe_n_yes": int((y_true == 1).sum()),
            "probe_n_no": int((y_true == 0).sum()),
            "wikitext_path": str(WIKIPQ), "wikitext_sha256_observed": wiki_sha,
            "yes_id": int(yes_id), "no_id": int(no_id),
        },
        "perplexity": {
            "n_paragraphs_used": int(n_para_used),
            "corpus_ppl": {a: round(float(ppl[a]), 4) for a in ppl_arms},
            "pct_change_vs_clean": {a: round(float(ppl_pct[a]), 4) for a in ppl_arms if a != "clean"},
            "tox_fire_rate_per_1k_tokens": tox_fire_per_1k,
            "note": "token-weighted corpus PPL = exp(sum NLL / sum tokens)",
        },
        "probe": {
            "locus_gate": {"base_argmax_in_yesno_frac": round(locus_frac, 4), "threshold": 0.95,
                           "pass": locus_pass},
            "metrics": probe_metrics,
            "bootstrap_ci95": boot,
        },
        "interpretation": interpretation,
    }

    # ---- GATE evaluation --------------------------------------------------------------------------
    # mechanical gates (nonzero exit on FAIL)
    n_q = len(questions)
    shapes_ok = (len(base_gap) == n_q and all(len(gaps[a]) == n_q for a in gaps)
                 and rand_gaps.shape == (R_RAND_PROBE, n_q) and len(ff_base) == n_q
                 and len(ff_tox) == n_q and all(len(nll_store[a]) == n_para_used for a in ppl_arms))
    sha_ok = bool(probe_sha) and bool(wiki_sha)

    # scientific gates (exit 0 regardless)
    tox_pct = ppl_pct["tox"]
    worst_control_pct = max(ppl_pct["decmatch"], ppl_pct["random"])
    g_ppl_within = bool(abs(tox_pct) <= acceptance_criteria["ppl_tox_within_pct"])
    g_ppl_notworse = bool(tox_pct <= worst_control_pct + acceptance_criteria["ppl_materiality_margin_pct"])
    acc_drop_pp = (acc_base - acc(gaps["tox"])) * 100.0
    ci = boot["acc_base_minus_acc_tox"]["ci95"]
    ci_contains_zero = bool(ci[0] <= 0.0 <= ci[1])
    g_probe_acc = bool(acc_drop_pp <= acceptance_criteria["probe_acc_drop_soft_pp"] and ci_contains_zero
                       and acc_drop_pp <= acceptance_criteria["probe_acc_drop_hard_pp"])
    g_probe_flip = bool(flip(gaps["tox"]) <= acceptance_criteria["probe_flip_max"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[cap] wrote {out_path}", flush=True)

    mechanical_fail = False
    def gate(name, ok, detail, mechanical=False):
        nonlocal mechanical_fail
        print(f"GATE {name} {'PASS' if ok else 'FAIL'} {detail}", flush=True)
        if mechanical and not ok:
            mechanical_fail = True

    gate("locus", locus_pass, f"base_argmax_in_yesno={locus_frac:.4f} >= 0.95", mechanical=True)
    gate("probe_balance", balance_ok,
         f"n={n_q} yes={bal_yes} no={bal_no} expected {exp_n}/{exp_yes}/{exp_n - exp_yes}", mechanical=True)
    gate("shapes", shapes_ok, f"probe_n={n_q} ppl_paras={n_para_used} arms_consistent", mechanical=True)
    gate("sha_recorded", sha_ok, f"probe_sha={probe_sha[:12]} wiki_sha={wiki_sha[:12]}", mechanical=True)
    # scientific
    gate("ppl_tox_within5", g_ppl_within, f"pct_change(tox)={tox_pct:.3f} within +/-5%")
    gate("ppl_not_worse_than_controls", g_ppl_notworse,
         f"tox={tox_pct:.3f} vs worst(decmatch,random)={worst_control_pct:.3f} margin<=1pp")
    gate("probe_acc_drop", g_probe_acc,
         f"acc_base={acc_base:.4f} acc_tox={acc(gaps['tox']):.4f} drop={acc_drop_pp:.2f}pp "
         f"ci={ci} contains0={ci_contains_zero} (soft<=2pp,hard<=5pp)")
    gate("probe_flip", g_probe_flip, f"flip(tox)={flip(gaps['tox']):.4f} <= 0.05")

    print(f"[cap] SUMMARY ppl_tox_pct={tox_pct:.3f} acc_base={acc_base:.4f} acc_tox={acc(gaps['tox']):.4f} "
          f"flip_tox={flip(gaps['tox']):.4f} flip_rand={rand_flip:.4f} "
          f"tox_fire/1k={tox_fire_per_1k['total_all_layers']}", flush=True)

    if mechanical_fail:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="20 paragraphs + 20 questions -> *_smoke.json")
    a = ap.parse_args()
    run(smoke=a.smoke)


if __name__ == "__main__":
    main()
