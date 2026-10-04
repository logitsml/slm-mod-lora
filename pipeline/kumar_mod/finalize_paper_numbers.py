"""Stamp the canonical paper numbers. Reads the
result JSONs and writes results/kumar_mod/PAPER_NUMBERS.json (machine-readable) with provenance + a
timestamp passed in on argv (the script reads no clock, so the stamp is reproducible).

Run: uv run python -m pipeline.kumar_mod.finalize_paper_numbers "<iso-timestamp>"
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BAL = ROOT / "results" / "kumar_mod" / "balanced"
QW = ROOT / "results" / "kumar_mod" / "analysis"
KM = ROOT / "results" / "kumar_mod"


def _load(p):
    # Missing/partial result files must not abort the stamp: a run that lacks one
    # arm still yields a valid PAPER_NUMBERS.json with that field carrying _error.
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:120]}", "_path": str(p)}


def _sae_block():
    sc = _load(KM / "sae_causal_dosedense.json")
    su = _load(KM / "sae_sufficiency.json")
    inv = _load(KM / "sae" / "invariance_w16k.json")
    dla = _load(KM / "sae" / "dla_w16k.json")
    ec = _load(QW / "encoder_toxicity_collapse.json")
    eg = _load(QW / "encoder_head_geometry.json")
    trx = _load(QW / "toxicity_residualization.json")

    def r(x, n):
        # Round only real numbers; anything missing collapses to null so the
        # downstream JSON never carries a stale or type-confused value.
        return round(x, n) if isinstance(x, (int, float)) else None

    fr = (sc.get("flip_rate", {}) or {}).get("all", {})
    diss = sc.get("dissociation_ci95", {})
    dr = sc.get("dose_response_flip", {})
    reagg = sc.get("re_aggregation", {})
    sir = su.get("induction_rate", {})
    sdr = su.get("dose_response_induction", {})
    sdiss = su.get("dissociation_ci95", {})
    # Invariance is read at L31 under the top-K20 consensus set; DLA share at top-K50.
    # The K values pin which feature budget the paper quotes for each diagnostic.
    invk = ((inv.get("L31", {}) or {}).get("C3_consistency", {}) or {}).get("K20", {})
    d31 = (dla.get("L31", {}) or {}).get("K50", {}) or {}
    d41 = (dla.get("L41", {}) or {}).get("K50", {}) or {}
    ecd = ec.get("AUC_tox_to_decision", {})
    trbm = trx.get("by_method", {}) if isinstance(trx, dict) else {}

    def nontox(k):
        return r(((trbm.get(k, {}) or {}).get("nontoxic_removal_auc", {}) or {}).get("median"), 3)

    return {
        "_source": ("results/kumar_mod/{sae_causal_dosedense,sae_sufficiency}.json, "
                    "sae/{invariance,dla}_w16k.json, "
                    "analysis/{encoder_toxicity_collapse,encoder_head_geometry,toxicity_residualization}.json"),
        "model": "gemma-3-12b-it",
        "sae": "gemma-scope-2-12b-it resid_post 16k l0_medium L24/31/41",
        "necessity_flip_to_keep": {
            # Necessity: ablate frozen toxicity features and measure the rate at which
            # remove-decisions flip to keep. decmatch / community / random are the
            # matched control feature sets; the tox-minus-decmatch CI is the headline
            # dissociation (toxicity features carry the effect, not decision-correlated ones).
            "tox": r(fr.get("tox"), 4),
            "decision_matched_nontox": r(fr.get("decmatch"), 3),
            "community": r(fr.get("community"), 3),
            "random": r(fr.get("random"), 4),
            "tox_minus_decmatch_ci95": [r(v, 3) for v in
                                        ((diss.get("tox_minus_decmatch", {}) or {}).get("ci95") or [None, None])],
            "within_label_strata_tox": {
                "human_keep": r(((sc.get("flip_rate", {}) or {}).get("human_keep", {}) or {}).get("tox"), 4),
                "human_remove": r(((sc.get("flip_rate", {}) or {}).get("human_remove", {}) or {}).get("tox"), 4)},
            # Dose-response at steering strengths alpha=0.5 and 1.0 confirms the flip
            # scales monotonically with how hard the toxicity direction is ablated.
            "dose_alpha": {"0.5": r(dr.get("a0.5"), 3), "1.0": r(dr.get("a1.0"), 3)},
            # Re-aggregation contrast: adding L41 to the {L24,L31} ablation set vs not,
            # to show the effect is not an artifact of any single layer's features.
            "reaggr_L24_31_41_vs_L24_31": [r(reagg.get("tox_L24_31_41"), 3), r(reagg.get("tox_L24_31_only"), 3)]},
        "sufficiency_induce_removal": {
            # Sufficiency is the converse of necessity: inject the toxicity direction
            # into keep-decisions and measure induced removals. p95/p99 are the
            # high-percentile steering operating points the paper reports.
            "tox_p95": r(sir.get("tox_p95"), 3),
            "p99": r(sdr.get("p99"), 3),
            "community": r(sir.get("community_p95"), 3),
            "random": r(sir.get("random_p95"), 3),
            "tox_minus_random_ci95": [r(v, 3) for v in
                                      ((sdiss.get("tox_minus_random", {}) or {}).get("ci95") or [None, None])]},
        "invariance_cross_community_jaccard": {
            # Community-invariance: Jaccard overlap of the top toxicity features selected
            # independently per community. Compared against a shuffle null so the overlap
            # is read relative to chance, not in absolute terms.
            "L31": r(invk.get("median_jaccard"), 2),
            "shuffle_null": r(invk.get("null_median_jaccard_mean"), 2),
            "universal_set_tox_frac": r(invk.get("universal_set_tox_frac"), 2)},
        "dla_toxicity_share_readout": {
            "L31": r(d31.get("share_tox"), 2),
            "L41": r(d41.get("share_tox"), 3),
            "community": r(d41.get("share_comm"), 2),
            "lens_cert_L41": r((dla.get("L41", {}) or {}).get("cert_corr_lens_gap"), 3)},
        "encoder_contrast": {
            "auc_tox_to_decision": {"human": r(ecd.get("human"), 2), "encoder": r(ecd.get("encoder"), 2),
                                    "llm_gemma": r(ecd.get("llm_gemma"), 2)},
            # CI is the subreddit-clustered bootstrap (resample communities, not rows)
            # so the interval respects within-community correlation rather than treating
            # every comment as independent.
            "llm_minus_encoder_ci95": [r(v, 3) for v in
                                       ((ec.get("contrast_llm_minus_encoder", {}) or {}).get("ci95_subreddit_bootstrap")
                                        or [None, None])],
            "head_cross_community_cos": r((eg.get("cross_head_abscos", {}) or {}).get("mean"), 2),
            "head_toxicity_alignment_cos": r((eg.get("head_toxicity_alignment_abscos", {}) or {}).get("detoxify_mean"), 2),
            "nontoxic_removal_auc": {"encoder_e5": nontox("encoder_e5"), "slm_mod": nontox("slm_mod"),
                                     "llm_gemma": nontox("llm_gemma")}},
        "claim_ceiling": ("A (causal toxicity reliance)=PROOF, gemma-only; B (community-invariance)=convergent; "
                          "encoder=behavioral+geometry"),
    }


def main():
    ts = sys.argv[1] if len(sys.argv) > 1 else "UNSTAMPED"
    fc = _load(BAL / "fairness_compare.json")
    tr = _load(QW / "toxicity_residualization.json")
    bm = fc.get("by_method", {}) if isinstance(fc, dict) else {}

    def method_row(k):
        m = bm.get(k, {})
        if not isinstance(m, dict) or "bal_auc" not in m:
            return {"status": m.get("status", "absent") if isinstance(m, dict) else "absent"}
        # Held-out vs in-sample certification: the *_heldout fraction certifies on the
        # 20% test split and is the quotable number; the in-sample variant is optimistic
        # (same data used to pick the threshold) and is kept only as a diagnostic.
        return {"bal_auc_median": (m.get("bal_auc") or {}).get("median"),
                "pr_auc_median": (m.get("pr_auc") or {}).get("median"),
                "cert95_heldout_PAPER": m.get("frac_subs_certify_0.95_precision_heldout"),
                "cert90_heldout_PAPER": m.get("frac_subs_certify_0.90_precision_heldout"),
                "cert95_in_sample_DO_NOT_QUOTE": m.get("frac_subs_certify_0.95_precision"),
                "cost": m.get("cost")}

    paper = {
        "CANONICAL_NUMBERS": True,
        "generated": ts,
        "provenance": {
            "rule": "QUOTE the fields named *_heldout / *_PAPER; in-sample cert fields are diagnostic only.",
            "scoring": {
                "llm_zeroshot_*": ("rating-scored medians (secondary analysis); the paper's primary LLM "
                                   "BAL-AUC/PR-AUC are next-token yes/no logit-gap scored and live in "
                                   "llm_gap_primary_metrics.json (gap_primary_recompute.py)"),
                "encoder_e5 / slm_mod / tfidf": "scored by their native continuous outputs",
            },
        },
        "head_to_head_fairness_compare": {k: method_row(k) for k in
                                          ["encoder_e5", "slm_mod", "llm_zeroshot_gemma", "llm_zeroshot_llama",
                                           "llm_zeroshot_qwen", "tfidf"]},
        # AUC of toxicity predicting NON-toxic removals: how much each method moderates
        # rule-breaking-but-clean content on the basis of toxicity. High for the supervised
        # encoder/SLM (genuine non-toxic-norm signal); low/near-chance for the LLMs (the
        # collapse).
        "toxicity_collapse_nontoxic_removal_auc": (
            (lambda trbm: {k: ((trbm.get(k, {}) or {}).get("nontoxic_removal_auc", {}) or {}).get("median")
                           for k in ["encoder_e5", "slm_mod", "llm_gemma", "llm_llama", "llm_qwen"]})(
                tr.get("by_method", {}) if isinstance(tr, dict) else {})),
        # The toxicity-collapse keeper distinguishes two measurements that are easy to
        # conflate: the behavioral metric (toxicity predicts the model's own decisions)
        # leads; the representational cosine is corroborating. Lead with the behavioral
        # metric and reserve causal "toxicity collapse" for the gated steering ablation.
        "mechinterp_keeper": (lambda td: {
            "headline": "BEHAVIORAL (cross-family): toxicity predicts MODEL decisions better than HUMAN removals",
            "TC_behav_macro_would_moderate_by_family": {
                fam: ((((td.get("behavioral_by_family", {}) or {}).get(fam, {}) or {}).get("by_scorer", {}) or {})
                      .get("detoxify", {}).get("would_moderate", {}) or {}).get("TC_behav_macro")
                for fam in ("gemma", "llama", "qwen")},
            "TC_rep_model_cos_deconf_by_layer": {
                L: ((v.get("model", {}) or {}).get("cos_deconf"))
                for L, v in (td.get("representational_layer_sweep", {}) or {}).items()},
            "source": "results/kumar_mod/analysis/tc_deepdive.json (model-DECISION direction)",
        })(_load(QW / "tc_deepdive.json")),
        "sae_causal": _sae_block(),
    }
    out = KM / "PAPER_NUMBERS.json"
    # Merge with prior store: setdefault carries forward only a TOP-LEVEL key that the
    # current run does not emit at all. It does NOT protect a block whose inputs were
    # missing this run: every top-level block here is rebuilt unconditionally, so a
    # missing input yields null/_error values nested inside an always-present block,
    # which overwrites the prior good block. Freshly computed keys always win.
    if out.exists():
        prev = json.loads(out.read_text())
        for key, val in prev.items():
            paper.setdefault(key, val)
    out.write_text(json.dumps(paper, indent=2, default=str))
    print(f"[finalize] wrote {out}")
    print(json.dumps(paper["head_to_head_fairness_compare"], indent=2, default=str))


if __name__ == "__main__":
    main()
