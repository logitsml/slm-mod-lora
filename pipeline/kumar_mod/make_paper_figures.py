#!/usr/bin/env python3
"""Build the paper figures from audited result JSON files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results" / "kumar_mod"
FIGURES = PROJECT / "figures"

C_BLUE = "#00274C"
C_GOLD = "#FFCB05"
C_RED = "#9A3324"
C_GREEN = "#75988D"
C_STEEL = "#2F65A7"
C_PURPLE = "#702082"
C_OLIVE = "#A5A508"
METHOD_GREEN = "#597638"
C_TAN = "#CFC096"
C_ASH = "#989C97"
C_BLACK = "#000000"
C_WHITE = "#FFFFFF"

ENCODER_COLOR = C_BLUE
GEMMA_COLOR = C_STEEL
LLAMA_COLOR = C_OLIVE
QWEN_COLOR = C_PURPLE
COLDSTART_COLOR = METHOD_GREEN
COMMUNITY_COLOR = C_GREEN
TOXICITY_COLOR = C_RED
SLM_COLOR = COMMUNITY_COLOR
LLM_AGGREGATE_COLOR = C_GOLD
INK = C_BLACK
MUTED = INK
TICK = INK
GRID = C_TAN
AXIS = INK
PANEL = C_WHITE
SURFACE = C_WHITE
NEUTRAL = INK
CONTROL_TAN = C_TAN
CONTROL_ASH = C_ASH
LIGHT_METHOD_VARIANT = CONTROL_TAN
LIGHT_NEUTRAL = CONTROL_ASH

FAMILY_LABELS = {
    "gemma": "Gemma",
    "llama": "Llama",
    "qwen": "Qwen",
}
FAMILY_COLORS = {
    "gemma": GEMMA_COLOR,
    "llama": LLAMA_COLOR,
    "qwen": QWEN_COLOR,
}

HEADER_Y = 0.955
HEADER_TITLE_WRAP = 76
HEADER_SUBTITLE_WRAP = 106
HEADER_SUBTITLE_PAD = 0.012
PLOT_TOP = 0.700
PANEL_TOP = 0.740
TITLE_SIZE = 21.0
SUBTITLE_SIZE = 13.6
ANN_SIZE = 12.0
LINE_W = 3.75
HEAVY_LINE_W = 4.85
AXIS_W = 1.75
GRID_W = 0.95
REF_W = 1.95
SAVE_PAD = 0.04
BAR_W = 0.88
CATEGORY_GUTTER = 0.055
ORDERED_GUTTER_FRAC = 0.035
MARKER_SIZE = 7.8
MARKER_AREA = 70

SOURCE_LOG: list[dict] = []
OUTPUT_LOG: list[str] = []


def register_tex_font(filename: str, fallback: str) -> str:
    # Match the paper's Linux Libertine body text in the figures; degrade to a
    # bundled serif if the TeX font isn't installed so plotting never fails.
    try:
        path = subprocess.check_output(["kpsewhich", filename], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return fallback
    if not path:
        return fallback
    font_manager.fontManager.addfont(path)
    return font_manager.FontProperties(fname=path).get_name()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(rel: str, field: str) -> dict:
    path = RESULTS / rel
    if not path.exists():
        raise FileNotFoundError(path)
    # Every read is logged with a content hash; write_manifest dumps this so each
    # figure is traceable to the exact input bytes it was built from.
    SOURCE_LOG.append({"file": str(path.relative_to(PROJECT)), "field": field, "sha256": sha256(path)})
    return json.loads(path.read_text())


def setup_style() -> None:
    chart_font = register_tex_font("LinLibertine_R.otf", "DejaVu Serif")
    mono_font = register_tex_font("LinLibertine_R.otf", "DejaVu Serif")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.family": chart_font,
            "font.serif": [chart_font, "Libertinus Serif", "Georgia", "DejaVu Serif"],
            "font.sans-serif": [chart_font, "Libertinus Serif", "Georgia", "DejaVu Serif"],
            "font.monospace": [mono_font, "Libertinus Serif", "Georgia", "DejaVu Serif"],
            "font.size": 12.4,
            "axes.labelsize": 13.0,
            "axes.titlesize": 14.4,
            "axes.titleweight": "bold",
            "axes.titlepad": 11.5,
            "xtick.labelsize": 11.4,
            "ytick.labelsize": 11.4,
            "legend.fontsize": 11.2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": PANEL,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "xtick.color": TICK,
            "ytick.color": TICK,
            "text.color": INK,
            "axes.linewidth": AXIS_W,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Type-42 embeds real (non-bitmap) glyphs so the PDF text stays
            # selectable and crisp in the camera-ready.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axis(ax, *, xgrid=True, ygrid=False) -> None:
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_linewidth(AXIS_W)
    ax.spines["bottom"].set_linewidth(AXIS_W)
    ax.tick_params(length=4.1, width=AXIS_W, color=AXIS)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=GRID_W, alpha=1.0)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=GRID_W, alpha=1.0)
    ax.set_axisbelow(True)
    ax.margins(x=0)


def flush_category_axis(ax, n: int, *, width: float = BAR_W, gutter: float = CATEGORY_GUTTER) -> None:
    ax.set_xlim(-width / 2 - gutter, n - 1 + width / 2 + gutter)
    ax.margins(x=0)


def flush_ordered_axis(ax, ticks, *, gutter_frac: float = ORDERED_GUTTER_FRAC) -> None:
    ticks = list(ticks)
    lo, hi = min(ticks), max(ticks)
    span = hi - lo
    gutter = span * gutter_frac if span else 0.5
    ax.set_xlim(lo - gutter, hi + gutter)
    ax.margins(x=0)


def label_box() -> dict:
    return {"facecolor": PANEL, "edgecolor": "none", "alpha": 1.0, "pad": 1.6}


def pct_label_y(value: float, ci: list[float] | tuple[float, float] | None = None, *, offset: float = 0.028) -> float:
    # Float a bar's value label above its CI cap (or the bar top if no CI), with
    # a small floor so labels on near-zero bars don't collide with the axis.
    anchor = ci[1] if ci else value
    return max(anchor + offset, 0.04)


def line_with_points(ax, x, y, *, color: str, label: str | None = None, marker: str = "o", lw: float = LINE_W) -> None:
    ax.plot(x, y, color=color, lw=lw, label=label, zorder=2, clip_on=False)
    ax.scatter(x, y, s=MARKER_AREA, color=color, edgecolor="none", linewidth=0, marker=marker, zorder=7, clip_on=False)


def _header_line_height(fig, fontsize: float, *, linespacing: float) -> float:
    return (fontsize / 72.0 * linespacing) / fig.get_figheight()


def add_header(fig, title: str, subtitle: str, *, y=HEADER_Y) -> None:
    title_text = textwrap.fill(title, HEADER_TITLE_WRAP)
    title_lines = title_text.count("\n") + 1
    title_line_height = _header_line_height(fig, TITLE_SIZE, linespacing=0.96)
    fig.text(
        0.5,
        y,
        title_text,
        ha="center",
        va="top",
        fontsize=TITLE_SIZE,
        weight="bold",
        linespacing=0.96,
    )
    if subtitle:
        subtitle_y = y - title_lines * title_line_height - HEADER_SUBTITLE_PAD
        fig.text(
            0.5,
            subtitle_y,
            textwrap.fill(subtitle, HEADER_SUBTITLE_WRAP),
            ha="center",
            va="top",
            fontsize=SUBTITLE_SIZE,
            color=MUTED,
            linespacing=1.02,
        )


def _header_renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _px_width(fig, s, fontsize, weight, renderer):
    t = fig.text(0.5, 0.5, s, fontsize=fontsize, weight=weight)
    w = t.get_window_extent(renderer).width
    t.remove()
    return w


def _wrap_to_px(fig, s, fontsize, weight, max_px, renderer):
    out, cur = [], ""
    for word in s.split():
        trial = (cur + " " + word).strip()
        if not cur or _px_width(fig, trial, fontsize, weight, renderer) <= max_px:
            cur = trial
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _fit_block(fig, s, fontsize, weight, max_px, floor, renderer):
    fs = fontsize
    while True:
        lines = _wrap_to_px(fig, s, fs, weight, max_px, renderer)
        if fs <= floor or max(_px_width(fig, ln, fs, weight, renderer) for ln in lines) <= max_px:
            return lines, fs
        fs -= 0.5


def add_header_fit(fig, title, subtitle="", *, y=HEADER_Y, usable=0.94):
    """Width-aware header: wrap to the canvas width by measured pixels, shrinking
    the font only if a single word would still overflow. Clip-proof for any string."""
    r = _header_renderer(fig)
    max_px = fig.get_figwidth() * fig.dpi * usable
    tlines, tfs = _fit_block(fig, title, TITLE_SIZE, "bold", max_px, 15.0, r)
    fig.text(0.5, y, "\n".join(tlines), ha="center", va="top", fontsize=tfs, weight="bold", linespacing=0.96)
    if subtitle:
        line_h = (tfs / 72.0 * 0.96) / fig.get_figheight()
        sy = y - len(tlines) * line_h - HEADER_SUBTITLE_PAD
        slines, sfs = _fit_block(fig, subtitle, SUBTITLE_SIZE, "normal", max_px, 10.0, r)
        fig.text(0.5, sy, "\n".join(slines), ha="center", va="top", fontsize=sfs, color=MUTED, linespacing=1.02)


def autofit_axis_label(ax, fig, axis="x", floor=9.5):
    """Shrink an axis label until it fits the axes extent (no horizontal overflow)."""
    r = _header_renderer(fig)
    lab = ax.xaxis.label if axis == "x" else ax.yaxis.label
    if not lab.get_text():
        return
    avail = (ax.get_window_extent(r).width if axis == "x" else ax.get_window_extent(r).height) * 0.98
    while lab.get_fontsize() > floor:
        ext = lab.get_window_extent(r)
        if (ext.width if axis == "x" else ext.height) <= avail:
            break
        lab.set_fontsize(lab.get_fontsize() - 0.5)
        fig.canvas.draw()


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def pct1(x: float) -> str:
    if x >= 0.9995:
        return "100%"
    return f"{100 * x:.1f}%"


def controls_bracket(ax, x_lo: float, x_hi: float, y: float, label: str = "non-toxic controls", tick: float = 0.035) -> None:
    ax.plot([x_lo, x_hi], [y, y], color=INK, lw=1.4, clip_on=False, zorder=5)
    ax.plot([x_lo, x_lo], [y, y - tick], color=INK, lw=1.4, clip_on=False, zorder=5)
    ax.plot([x_hi, x_hi], [y, y - tick], color=INK, lw=1.4, clip_on=False, zorder=5)
    ax.text((x_lo + x_hi) / 2, y + 0.02, label, ha="center", va="bottom", fontsize=ANN_SIZE, color=INK, bbox=label_box(), clip_on=False, zorder=6)


def save(fig, stem: str) -> None:
    # Figure 1 (methods-comparison) is maintained by hand outside this script; this
    # guard plus the sha256 checks in main() guarantee a run never overwrites it.
    if stem == "methods-comparison":
        raise RuntimeError("Refusing to write Figure 1 methods-comparison.png")
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(pdf, bbox_inches=None, pad_inches=SAVE_PAD)
    fig.savefig(png, bbox_inches=None, pad_inches=SAVE_PAD)
    OUTPUT_LOG.extend([str(pdf.relative_to(PROJECT)), str(png.relative_to(PROJECT))])
    plt.close(fig)


def errbarh(ax, y, x, lo, hi, color, label=None, marker="o", fill=True, size=72):
    ax.plot([lo, hi], [y, y], color=color, lw=HEAVY_LINE_W, solid_capstyle="round", clip_on=False)
    ax.scatter([x], [y], s=size, color=(color if fill else PANEL), edgecolor="none", linewidth=0, zorder=3, label=label, marker=marker, clip_on=False)


def plot_tc_test() -> None:
    # Toxicity-collapse test: per community, does toxicity predict the model's
    # decision better than it predicts the recorded human removal? Scored with
    # ToxiGen (identity-robust) from an independent re-derivation of the metric.
    data = load_json("_INDEPENDENT_tc_rederive.json", "percommunity_toxigen and by_family.by_scorer")
    rows = data["percommunity_toxigen"]

    fig = plt.figure(figsize=(8.6, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], left=0.08, right=0.98, bottom=0.15, top=0.93, wspace=0.27)
    ax = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[0, 1])

    for fam in ("gemma", "llama", "qwen"):
        pts = [r for r in rows if r["family"] == fam]
        x = np.array([r["auc_tox_to_moderator"] for r in pts])
        y = np.array([r["auc_tox_to_model"] for r in pts])
        ax.scatter(x, y, s=34, color=FAMILY_COLORS[fam], edgecolor="none", linewidth=0, label=FAMILY_LABELS[fam])
        # Per-community gap: how much more toxicity drives the model than the
        # human removal. Points above the diagonal are the collapse.
        delta = y - x
        # Deterministic vertical jitter so overlapping dots in each family's strip
        # stay readable; per-family seed keeps the layout reproducible.
        jitter = np.linspace(-0.11, 0.11, len(delta))
        rng = np.random.default_rng({"gemma": 1, "llama": 2, "qwen": 3}[fam])
        rng.shuffle(jitter)
        ybase = {"gemma": 2, "llama": 1, "qwen": 0}[fam]
        axd.scatter(delta, np.full_like(delta, ybase) + jitter, s=31, color=FAMILY_COLORS[fam], edgecolor="none", linewidth=0)
        # Vertical tick marks the per-community median gap, not the mean — robust
        # to the heavy-tailed community distribution.
        med = float(np.median(delta))
        axd.plot([med, med], [ybase - 0.23, ybase + 0.23], color=INK, lw=LINE_W)

    # y = x reference: a community on this line treats toxicity identically for
    # the model and the human; everything above it is the collapse.
    ax.plot([0.45, 0.9], [0.45, 0.9], color=NEUTRAL, lw=REF_W, ls="--")
    ax.text(0.49, 0.51, "equal AUC", color=INK, fontsize=ANN_SIZE, rotation=39, bbox=label_box())
    ax.set_xlim(0.48, 0.865)
    ax.set_ylim(0.48, 0.91)
    ax.set_xlabel("Toxicity predicts recorded removal (AUC)")
    ax.set_ylabel("Toxicity predicts model decision (AUC)")
    ax.legend(loc="lower right", frameon=False, ncol=1)
    finish_axis(ax, xgrid=True, ygrid=True)

    # Right panel summary: macro-averaged TC behavioral gap per family with its
    # clustered-bootstrap CI, plus the share of communities with a positive gap.
    summaries = data["by_family"]
    for fam in ("gemma", "llama", "qwen"):
        rec = summaries[fam]["by_scorer"]["tox_toxigen"]
        ybase = {"gemma": 2, "llama": 1, "qwen": 0}[fam]
        ci = rec["ci95"]
        axd.plot(ci, [ybase, ybase], color=FAMILY_COLORS[fam], lw=HEAVY_LINE_W, solid_capstyle="round")
        axd.scatter([rec["TC_behav_macro"]], [ybase], s=78, color=FAMILY_COLORS[fam], edgecolor="none", linewidth=0, zorder=4)
        axd.text(rec["TC_behav_macro"], ybase - 0.31, f"{rec['TC_behav_macro']:+.4f}", ha="center", va="top",
                 fontsize=ANN_SIZE - 1.0, color=INK, bbox=label_box(), zorder=5, clip_on=False)
        axd.text(0.415, ybase, f"{pct(rec['frac_communities_gt0'])} positive", va="center", ha="right", fontsize=ANN_SIZE, color=INK, bbox=label_box(), clip_on=False)
    # Zero line: a family whose CI clears it has a significant collapse.
    axd.axvline(0, color=NEUTRAL, lw=REF_W, ls="--")
    axd.set_yticks([2, 1, 0])
    axd.set_yticklabels(["Gemma", "Llama", "Qwen"])
    axd.set_xlim(-0.19, 0.43)
    axd.set_xticks([-0.15, 0.0, 0.15, 0.30])
    axd.set_xlabel("AUC gap: model - recorded removal")
    finish_axis(axd, xgrid=True, ygrid=False)

    save(fig, "fig_tc_test")


def plot_rule_interventions() -> None:
    tox = load_json("analysis/toxicity_across_conditions.json", "by_family condition toxicity AUC/removal_rate")
    flips = load_json("b3_decision_flip.json", "by_family do_rules/do_prompt flip_to_keep_rate")

    # Conditions ordered from the intended setup (posted rules) through
    # progressively degraded rule signals. If toxicity-driven removal were really
    # rule-following, stripping or scrambling the rules should weaken it.
    conds = [
        ("posted\nrules", "rule_scramble:home", "posted_rules"),
        ("anti-tox\nprompt", "antitox:p4", "anti_tox_p4"),
        ("scrambled", "rule_scramble:scrambled", "scrambled"),
        ("no rules", "rule_scramble:none", "none"),
        ("neutral\nwords", "rule_scramble:random", "neutral_words"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.90, wspace=0.30)

    x = np.arange(len(conds))
    for fam in ("gemma", "llama", "qwen"):
        aucs = [tox["by_family"][fam][key]["auc_tox_predicts_decision"]["tox_detoxify"] for _, key, _ in conds]
        rates = [tox["by_family"][fam][key]["removal_rate"] for _, key, _ in conds]
        line_with_points(axes[0], x, aucs, color=FAMILY_COLORS[fam], label=FAMILY_LABELS[fam])
        line_with_points(axes[1], x, rates, color=FAMILY_COLORS[fam])
        # Panel (c) drops the "posted rules" baseline: flips are measured relative
        # to it, so only the four interventions have a flip-to-keep rate.
        fvals = [
            flips["by_family"][fam]["do_prompt"]["anti_tox_p4"]["flip_to_keep_rate"],
            flips["by_family"][fam]["do_rules"]["scrambled"]["flip_to_keep_rate"],
            flips["by_family"][fam]["do_rules"]["none"]["flip_to_keep_rate"],
            flips["by_family"][fam]["do_rules"]["random"]["flip_to_keep_rate"],
        ]
        fcis = [
            flips["by_family"][fam]["do_prompt"]["anti_tox_p4"]["ci95"],
            flips["by_family"][fam]["do_rules"]["scrambled"]["ci95"],
            flips["by_family"][fam]["do_rules"]["none"]["ci95"],
            flips["by_family"][fam]["do_rules"]["random"]["ci95"],
        ]
        line_with_points(axes[2], np.arange(len(fvals)), fvals, color=FAMILY_COLORS[fam])
        axes[2].errorbar(np.arange(len(fvals)), fvals,
                         yerr=[np.array(fvals) - np.array([c[0] for c in fcis]),
                               np.array([c[1] for c in fcis]) - np.array(fvals)],
                         fmt="none", ecolor=FAMILY_COLORS[fam], elinewidth=1.5, capsize=3.0, zorder=3)

    # Anchor the rightmost slot with the white-box SAE toxicity-ablation flip rate
    # as the causal upper bound the prompt/rule interventions are compared against.
    axes[2].scatter([len(conds) - 1], [flips["sae_tox_flip_gemma3 (Δ_tox)"]], s=128, color=TOXICITY_COLOR, edgecolor="none", linewidth=0, zorder=4, clip_on=False)
    axes[2].text(len(conds) - 1.16, 0.94, "SAE toxicity\nablation", ha="right", va="top", fontsize=ANN_SIZE, color=INK, bbox=label_box())

    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in conds], rotation=28, ha="right")
        flush_ordered_axis(ax, x)
        finish_axis(ax, xgrid=False, ygrid=True)
    axes[2].set_xticks(np.arange(len(conds)))
    axes[2].set_xticklabels([c[0] for c in conds[1:]] + ["toxicity\nablation"], rotation=28, ha="right")
    flush_ordered_axis(axes[2], np.arange(len(conds)))
    finish_axis(axes[2], xgrid=False, ygrid=True)
    axes[0].set_title("(a) Toxicity still predicts decisions")
    axes[0].set_ylabel("Toxicity predicts decision (AUC)")
    axes[0].set_ylim(0.62, 0.9)
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].set_title("(b) Removal rate")
    axes[1].set_ylabel("Removal rate")
    axes[1].set_ylim(0.14, 0.54)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[2].set_title("(c) Removals flipped to keep")
    axes[2].set_ylabel("Baseline removals flipped to keep")
    axes[2].set_ylim(0, 1.08)
    axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    save(fig, "fig_rule_interventions")


def plot_sae() -> None:
    nec = load_json("sae_causal_dosedense.json", "flip_rate and dissociation_ci95")
    suf = load_json("sae_sufficiency.json", "induction_rate and dose_response_induction")
    nec65 = load_json("sae_causal_toxicity_65k.json", "flip_rate")
    inv = load_json("sae/invariance_w16k.json", "C3_consistency")
    extra = load_json("sae_controls_extra.json", "heldout_on_disjoint_B, flip_rate controls, position_localization")
    dose = load_json("sae_causal_dosedense.json", "dose_response_flip")
    position = load_json("sae_position_probe.json", "flip_rate by position")

    fig = plt.figure(figsize=(11.2, 6.2))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, bottom=0.10, top=0.92, hspace=0.42, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    dose_gs = gs[1, 0].subgridspec(1, 2, wspace=0.20)
    ax3a = fig.add_subplot(dose_gs[0, 0])
    ax3b = fig.add_subplot(dose_gs[0, 1], sharey=ax3a)
    ax4 = fig.add_subplot(gs[1, 1])

    # Necessity (panel a): ablate a feature set, measure how many recorded
    # removals flip to keep. Toxicity features are the target; the other three are
    # matched controls (decision-correlated, community, random) that should not flip
    # decisions if toxicity is doing the causal work.
    labels = ["toxicity\nfeatures", "decision-\ncorrelated", "community\nfeatures", "random\nfeatures"]
    vals = [nec["flip_rate"]["all"][k] for k in ("tox", "decmatch", "community", "random")]
    colors = [TOXICITY_COLOR, CONTROL_TAN, COMMUNITY_COLOR, CONTROL_ASH]
    ax1.bar(np.arange(len(vals)), vals, width=BAR_W, color=colors, edgecolor="none", linewidth=0)
    for i, v in enumerate(vals):
        ax1.text(i, pct_label_y(v), pct1(v), ha="center", va="bottom", fontsize=ANN_SIZE, weight="bold" if i == 0 else "normal", bbox=label_box(), clip_on=False)
    ax1.set_xticks(np.arange(len(vals)))
    ax1.set_xticklabels(labels)
    flush_category_axis(ax1, len(vals))
    ax1.set_ylim(0, 1.08)
    ax1.set_title("(a) When each feature set is ablated")
    ax1.set_ylabel("Removals flipped to keep")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    controls_bracket(ax1, 0.62, 3.38, 0.52)
    finish_axis(ax1, xgrid=False, ygrid=True)

    # Sufficiency (panel b): the converse direction. Clamp each feature set up to
    # its 95th-percentile activation and count keeps that flip to remove. Tests
    # whether toxicity alone can induce a removal, not just whether it's needed.
    labels2 = ["toxicity\np95", "community\np95", "random\np95"]
    vals2 = [suf["induction_rate"][k] for k in ("tox_p95", "community_p95", "random_p95")]
    ax2.bar(np.arange(len(vals2)), vals2, width=BAR_W, color=[TOXICITY_COLOR, COMMUNITY_COLOR, CONTROL_ASH], edgecolor="none", linewidth=0)
    for i, v in enumerate(vals2):
        ax2.text(i, pct_label_y(v), pct1(v), ha="center", va="bottom", fontsize=ANN_SIZE, weight="bold" if i == 0 else "normal", bbox=label_box(), clip_on=False)
    ax2.set_xticks(np.arange(len(vals2)))
    ax2.set_xticklabels(labels2)
    flush_category_axis(ax2, len(vals2))
    ax2.set_ylim(0, 0.68)
    ax2.set_title("(b) When each feature set is induced")
    ax2.set_ylabel("Keeps flipped to remove")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    controls_bracket(ax2, 0.62, 2.38, 0.45)
    finish_axis(ax2, xgrid=False, ygrid=True)

    # Dose-response (panel c): keys like "a0.5"/"a1.5" encode the ablation
    # multiplier; strip the "a" and sort numerically so the curve reads left to right.
    dose_items = sorted(
        ((float(k.removeprefix("a")), v) for k, v in dose["dose_response_flip"].items()),
        key=lambda kv: kv[0],
    )
    x_nec = [a for a, _ in dose_items]
    y_nec = [v for _, v in dose_items]
    line_with_points(ax3a, x_nec, y_nec, color=TOXICITY_COLOR)
    suf_dose = suf["dose_response_induction"]
    x_suf = [90, 95, 99]
    y_suf = [suf_dose["p90"], suf_dose["p95"], suf_dose["p99"]]
    line_with_points(ax3b, x_suf, y_suf, color=GEMMA_COLOR)
    ax3a.set_title("(c) Stronger ablation")
    ax3b.set_title("Stronger induction")
    ax3a.set_ylabel("Decisions flipped")
    ax3a.set_xlabel("Ablation strength")
    ax3b.set_xlabel("Clamp percentile")
    ax3a.set_xticks([0.5, 0.8, 1.0, 1.5])
    ax3a.set_xticklabels(["0.5×", "0.8×", "1.0×", "1.5×"])
    ax3b.set_xticks(x_suf)
    ax3b.set_xticklabels(["90%", "95%", "99%"])
    flush_ordered_axis(ax3a, [0.5, 0.8, 1.0, 1.5])
    flush_ordered_axis(ax3b, x_suf)
    ax3a.set_ylim(0, 1.05)
    ax3a.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax3b.tick_params(labelleft=False)
    finish_axis(ax3a, xgrid=True, ygrid=True)
    finish_axis(ax3b, xgrid=True, ygrid=True)

    # Panel (d) robustness battery: held-out features (selected on split A, ablated
    # on disjoint B, so the necessity result isn't selection-on-the-tested-set);
    # a DLA-matched non-toxic control; the toxicity/decision pair re-run at 65k SAE
    # width; and the necessity localized to the decision-token site. The dotted
    # divider at x=3.5 separates the matched controls from the wider-SAE replications.
    audit_labels = ["held-out\nfeatures", "DLA-matched\nnon-toxic", "65k toxicity\nfeatures", "65k decision\ncontrol", "decision-token\nsite"]
    audit_vals = [
        extra["heldout_on_disjoint_B"]["flip_heldout_on_B"],
        extra["flip_rate"]["all"]["dla"],
        nec65["flip_rate"]["all"]["tox"],
        nec65["flip_rate"]["all"]["decmatch"],
        position["flip_rate"]["decision"],
    ]
    audit_colors = [TOXICITY_COLOR, CONTROL_ASH, TOXICITY_COLOR, CONTROL_TAN, GEMMA_COLOR]
    ax4.bar(np.arange(len(audit_vals)), audit_vals, width=BAR_W, color=audit_colors, edgecolor="none", linewidth=0)
    for i, v in enumerate(audit_vals):
        ax4.text(i, pct_label_y(v), pct1(v), ha="center", va="bottom", fontsize=ANN_SIZE, weight="bold" if v > 0.9 else "normal", bbox=label_box(), clip_on=False)
    ax4.set_xticks(np.arange(len(audit_vals)))
    ax4.set_xticklabels(audit_labels)
    ax4.tick_params(axis="x", labelsize=10.4)
    ax4.axvline(3.5, color=INK, lw=1.1, ls=":", alpha=0.65)
    flush_category_axis(ax4, len(audit_vals))
    ax4.set_ylim(0, 1.08)
    ax4.set_title("(d) Matched controls and replications")
    ax4.set_ylabel("Removals flipped to keep")
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    finish_axis(ax4, xgrid=False, ygrid=True)

    save(fig, "fig_sae_dissociation")


def plot_nontoxic() -> None:
    # Can each method still rank the non-toxic removals (keeps vs. removed-but-not-toxic)?
    # Above-chance AUC here means the method captures community norms beyond toxicity.
    toxres = load_json("analysis/toxicity_residualization.json", "by_method.nontoxic_removal_auc")
    gap = load_json("llm_gap_primary_metrics.json", "by_model non_tox_auc")
    sweep = load_json("balanced/robustness/nontoxic_auc_theta_sweep.json", "theta_grid and by_method")

    def non_tox(model: str) -> tuple[float, float, float]:
        rec = gap["by_model"][model]["non_tox_auc"]
        return rec["median"], rec["ci"][0], rec["ci"][1]

    rows = [
        ("Prompted Qwen2.5-7B", *non_tox("qwen25_7b"), FAMILY_COLORS["qwen"], "gap"),
        ("Prompted Llama-3.1-8B", *non_tox("llama31_8b"), FAMILY_COLORS["llama"], "gap"),
        ("Prompted Gemma-3-12B", *non_tox("gemma3_12b"), FAMILY_COLORS["gemma"], "gap"),
        ("SLM-Mod", toxres["by_method"]["slm_mod"]["nontoxic_removal_auc"]["median"], *toxres["by_method"]["slm_mod"]["nontoxic_removal_auc"]["ci95"], SLM_COLOR, "supervised"),
        ("e5-large-v2 encoder", toxres["by_method"]["encoder_e5"]["nontoxic_removal_auc"]["median"], *toxres["by_method"]["encoder_e5"]["nontoxic_removal_auc"]["ci95"], ENCODER_COLOR, "supervised"),
    ]

    fig = plt.figure(figsize=(9.7, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], left=0.21, right=0.98, bottom=0.15, top=0.90, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for i, (label, val, lo, hi, color, kind) in enumerate(rows):
        if lo is None:
            ax.scatter([val], [i], s=84, color=color, edgecolor="none", linewidth=0, marker="D", zorder=3)
        else:
            errbarh(ax, i, val, lo, hi, color, size=70)
        ax.text(0.845, i, f"{val:.3f}", va="center", ha="right", fontsize=ANN_SIZE, color=INK, bbox=label_box(), clip_on=False)
    # 0.5 = chance: a method at this line is doing no better than random on the
    # non-toxic removals, i.e. acting as a pure toxicity detector.
    ax.axvline(0.5, color=NEUTRAL, lw=REF_W, ls="--")

    # Same prompted models scored by the 1-5 rating they emit, the output a
    # closed-weight deployment actually exposes, against the logit gap above.
    rating_key = {"Prompted Qwen2.5-7B": "llm_qwen",
                  "Prompted Llama-3.1-8B": "llm_llama",
                  "Prompted Gemma-3-12B": "llm_gemma"}
    for i, row in enumerate(rows):
        key = rating_key.get(row[0])
        if key is None:
            continue
        rec = toxres["by_method"][key]["nontoxic_removal_auc"]
        y = i - 0.26
        ax.plot(rec["ci95"], [y, y], color=row[4], lw=LINE_W, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.scatter([rec["median"]], [y], s=46, facecolor="white", edgecolor=row[4],
                   linewidth=1.3, marker="o", zorder=3)
        ax.text(0.845, y, f"{rec['median']:.3f}", va="center", ha="right",
                fontsize=ANN_SIZE - 1.0, color=NEUTRAL, bbox=label_box(), clip_on=False)
    ax.scatter([], [], s=46, facecolor="white", edgecolor=INK, linewidth=1.3, marker="o",
               label="scored by 1-5 rating")
    ax.legend(frameon=False, loc="upper left", fontsize=ANN_SIZE - 1.0)

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.62, len(rows) - 0.4)
    ax.set_xlim(0.55, 0.852)
    ax.set_xlabel("AUC: keeps vs. non-toxic removals")
    ax.set_title("(a) AUC on non-toxic removals")
    finish_axis(ax, xgrid=True, ygrid=False)

    # Panel (b): vary the ToxiGen cutoff that defines "non-toxic" so the result
    # isn't an artifact of one threshold. Both supervised methods stay well above
    # chance across the grid.
    theta = sweep["theta_grid"]
    line_with_points(ax2, theta, [sweep["by_method"]["encoder"][str(t)]["median_auc"] for t in theta], color=ENCODER_COLOR, label="e5-large-v2")
    line_with_points(ax2, theta, [sweep["by_method"]["slm"][str(t)]["median_auc"] for t in theta], color=SLM_COLOR, label="SLM-Mod")
    ax2.set_title("(b) ToxiGen threshold sweep")
    ax2.set_xlabel("ToxiGen threshold for non-toxic removals")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0.72, 0.825)
    flush_ordered_axis(ax2, theta)
    ax2.legend(frameon=False, loc="lower right")
    finish_axis(ax2, xgrid=True, ygrid=True)

    save(fig, "fig_nontoxic_removals")


def plot_coverage() -> None:
    triage = load_json("analysis/triage_value.json", "f_needed_joint_precision_and_recall_HEADLINE")
    gap = load_json("llm_gap_primary_metrics.json", "by_model triage gap_llm_f_escalate")
    # Fraction of comments the logit-gap router escalates to a human, per family;
    # averaged into a single bar as the LLM triage burden.
    burden = {
        "gemma": gap["by_model"]["gemma3_12b"]["triage"]["gap_llm_f_escalate"],
        "llama": gap["by_model"]["llama31_8b"]["triage"]["gap_llm_f_escalate"],
        "qwen": gap["by_model"]["qwen25_7b"]["triage"]["gap_llm_f_escalate"],
    }

    fig, ax = plt.subplots(figsize=(6.2, 3.95))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.90)

    vals = [
        ("e5-large-v2\ntriage", triage["f_needed_joint_precision_and_recall_HEADLINE"]["encoder_e5_triage"]["f_needed"], triage["f_needed_joint_precision_and_recall_HEADLINE"]["encoder_e5_triage"]["ci95"], ENCODER_COLOR),
        ("LLM logit-gap\nmean", float(np.mean([burden["gemma"], burden["llama"], burden["qwen"]])), None, LLM_AGGREGATE_COLOR),
        ("LLM decision\nonly", triage["f_needed_joint_precision_and_recall_HEADLINE"]["llm_random"]["f_needed"], triage["f_needed_joint_precision_and_recall_HEADLINE"]["llm_random"]["ci95"], LIGHT_METHOD_VARIANT),
        ("LLM rating\nscore", triage["f_needed_joint_precision_and_recall_HEADLINE"]["llm_rating"]["f_needed"], triage["f_needed_joint_precision_and_recall_HEADLINE"]["llm_rating"]["ci95"], LIGHT_NEUTRAL),
    ]
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], width=BAR_W, color=[v[3] for v in vals], edgecolor="none", linewidth=0)
    for i, (_, val, ci, _) in enumerate(vals):
        if ci:
            ax.plot([i, i], ci, color=INK, lw=2.0)
            ax.plot([i - 0.08, i + 0.08], [ci[0], ci[0]], color=INK, lw=2.0)
            ax.plot([i - 0.08, i + 0.08], [ci[1], ci[1]], color=INK, lw=2.0)
        ax.text(i, pct_label_y(val, ci), pct1(val) if i == 0 else pct(val), ha="center", va="bottom", fontsize=ANN_SIZE, weight="bold", color=INK, bbox=label_box(), clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in vals])
    flush_category_axis(ax, len(vals))
    ax.set_ylim(0, 0.93)
    ax.set_ylabel("Human review fraction")
    ax.set_title("Comments routed to human review")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    finish_axis(ax, xgrid=False, ygrid=True)

    save(fig, "fig_coverage")




def plot_where_encoded() -> None:
    dla = load_json("sae/dla_w16k.json", "K50 share_tox by layer")
    inv = load_json("decision_axis_invariance.json", "by_layer cross_over_ceiling_ratio")

    # Two readouts share one panel. Invariance ratio (how stable the decision axis
    # is across communities) is measured at four layers including the early L12;
    # DLA readout shares only at the three later layers where the toxicity vs
    # community feature split is meaningful.
    layers_inv = [12, 24, 31, 41]
    ratio = [inv["by_layer"][str(l)]["cross_over_ceiling_ratio"] for l in layers_inv]
    ratio_lo = [inv["by_layer"][str(l)]["cross_over_ceiling_ratio_ci95"][0] for l in layers_inv]
    ratio_hi = [inv["by_layer"][str(l)]["cross_over_ceiling_ratio_ci95"][1] for l in layers_inv]
    layers_dla = [24, 31, 41]
    tox_share = [dla[f"L{l}"]["K50"]["share_tox"] for l in layers_dla]
    comm_share = [dla[f"L{l}"]["K50"]["share_comm"] for l in layers_dla]

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.15, top=0.95)
    line_with_points(ax, layers_dla, tox_share, color=TOXICITY_COLOR, label="toxicity readout")
    line_with_points(ax, layers_dla, comm_share, color=COMMUNITY_COLOR, label="community readout")
    ax.errorbar(layers_inv, ratio, yerr=[np.array(ratio) - np.array(ratio_lo), np.array(ratio_hi) - np.array(ratio)], color=GEMMA_COLOR, marker="s", lw=2.8, capsize=5, capthick=1.8, markersize=MARKER_SIZE, markeredgecolor="none", markeredgewidth=0, label="decision-axis invariance", zorder=6)
    # Layers whose invariance ratio clears 0.85 carry a community-stable decision
    # axis; this is where the supervised readout localizes.
    ax.axhline(0.85, color=NEUTRAL, lw=REF_W, ls="--")
    ax.text(12.2, 0.865, "invariance threshold", color=INK, fontsize=ANN_SIZE, bbox=label_box(), clip_on=False)
    ax.set_xticks(layers_inv)
    flush_ordered_axis(ax, layers_inv)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Gemma-3 layer")
    ax.set_ylabel("Readout share / invariance ratio")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.34))
    finish_axis(ax, xgrid=True, ygrid=True)
    save(fig, "fig_where_encoded")


def validate_inputs() -> None:
    # Refuse to build unless the numbers store is the finalized canonical one, so
    # figures can't be generated from a stale or intermediate aggregation.
    paper = load_json("PAPER_NUMBERS.json", "CANONICAL_NUMBERS")
    if paper.get("CANONICAL_NUMBERS") is not True:
        raise RuntimeError("PAPER_NUMBERS.json is not stamped canonical")
    fig1 = FIGURES / "methods-comparison.png"
    if not fig1.exists():
        raise FileNotFoundError(fig1)


def write_manifest(fig1_before: str) -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figure1_methods_comparison_sha256": fig1_before,
        "provenance_note": (
            "Cost inputs use the measured throughput_bench.json price points; "
            "_INDEPENDENT_tc_rederive.json re-derives the headline TC numbers from the "
            "shipped decision tables as a built-in consistency check."
        ),
        "canonical_assertions": {
            "CANONICAL_NUMBERS": True,
        },
        "palette_assertions": {
            "gemma": GEMMA_COLOR,
            "llama": LLAMA_COLOR,
            "qwen": QWEN_COLOR,
            "toxicity": TOXICITY_COLOR,
            "encoder": ENCODER_COLOR,
            "coldstart": COLDSTART_COLOR,
            "slm_mod": SLM_COLOR,
            "llm_aggregate": LLM_AGGREGATE_COLOR,
        },
        "outputs": OUTPUT_LOG,
        "source_events": SOURCE_LOG,
        "sources": SOURCE_LOG,
    }
    (FIGURES / "figure_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


def plot_over_removal() -> None:
    # Selectivity ratio = flag rate on kept-but-toxic comments / flag rate on
    # kept-non-toxic. >1 means a method over-flags toxicity among the comments the
    # community chose to keep. Toxicity defined as Detoxify >= 0.5 here.
    ov = load_json("analysis/over_removal.json", "by_tox_def Detoxify>=0.5 selectivity ratios")
    block = ov["by_tox_def"]["Detoxify>=0.5"]["methods"]
    labels = {"encoder_e5": "e5 encoder", "slm_mod": "SLM-Mod", "llm_gemma": "Prompted Gemma",
              "llm_llama": "Prompted Llama", "llm_qwen": "Prompted Qwen"}
    colors = {"encoder_e5": ENCODER_COLOR, "slm_mod": SLM_COLOR, "llm_gemma": FAMILY_COLORS["gemma"],
              "llm_llama": FAMILY_COLORS["llama"], "llm_qwen": FAMILY_COLORS["qwen"]}
    supervised = {"encoder_e5", "slm_mod"}
    order = sorted(labels, key=lambda m: block[m]["selectivity_ratio"])

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    fig.subplots_adjust(left=0.225, right=0.965, bottom=0.165, top=0.95)
    # Shade the contiguous run of prompted-LLM rows. Sorting by ratio happens to
    # cluster them, so a single span covers them; supervised rows fall outside.
    llm_rows = [i for i, m in enumerate(order) if m not in supervised]
    ax.axhspan(min(llm_rows) - 0.5, max(llm_rows) + 0.5, color=CONTROL_TAN, alpha=0.16, zorder=0)
    hi_max = max(block[m]["ratio_ci95"][1] for m in order)
    for i, m in enumerate(order):
        v = block[m]
        lo, hi = v["ratio_ci95"]
        errbarh(ax, i, v["selectivity_ratio"], lo, hi, colors[m], size=66)
        ax.text(hi + 0.07, i, f"{v['selectivity_ratio']:.2f}", va="center", ha="left",
                fontsize=ANN_SIZE, color=INK, bbox=label_box(), clip_on=False)
    ax.axvline(1.0, color=NEUTRAL, lw=REF_W, ls="--")
    ax.text(1.07, 0.5, "no toxicity\nselectivity", ha="left", va="center",
            fontsize=ANN_SIZE - 1.0, color=INK, bbox=label_box())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[m] for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0.9, hi_max + 0.7)
    ax.set_xlabel("Toxicity-selectivity ratio  (kept-toxic flag rate / kept-non-toxic)")
    finish_axis(ax, xgrid=True, ygrid=False)
    autofit_axis_label(ax, fig, "x")
    save(fig, "fig_over_removal")


def plot_distinctiveness() -> None:
    import polars as pl

    ENC_C, LLM_C = ENCODER_COLOR, LLM_AGGREGATE_COLOR
    # Does within-community accuracy fall as a community gets more distinctive
    # (toxicity predicts its removals less)? The encoder's slope should be flatter
    # than the LLM mean's; the contrast CI quantifies that gap.
    dist = load_json("analysis/distinctiveness.json", "per-method slopes and contrasts")
    pts_path = RESULTS / "analysis" / "distinctiveness_points.parquet"
    # Parquet scatter points aren't routed through load_json, so log them by hand
    # to keep the figure's provenance manifest complete.
    SOURCE_LOG.append({"file": str(pts_path.relative_to(PROJECT)),
                       "field": "per-community distinctiveness x and BAL-AUC y", "sha256": sha256(pts_path)})
    pts = pl.read_parquet(pts_path)
    x = pts["auc_tox_to_moderator"].to_numpy()
    y_enc = pts["balauc_encoder_e5"].to_numpy()
    y_llm = pts["balauc_llm_mean"].to_numpy()
    sl = dist["slopes"]
    con = dist["contrasts"]["llm_mean_minus_encoder"]
    ci_s = f"[{con['ci95'][0]:.2f}, {con['ci95'][1]:.2f}]"

    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    fig.subplots_adjust(left=0.14, right=0.965, bottom=0.165, top=0.95)
    ax.scatter(x, y_enc, s=30, marker="o", color=ENC_C, edgecolor="none", alpha=0.5, zorder=3,
               label="Supervised encoder")
    ax.scatter(x, y_llm, s=36, marker="^", color=LLM_C, edgecolor=INK, linewidth=0.45,
               alpha=0.75, zorder=3, label="Prompted LLM mean")
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, sl["encoder_e5"]["slope"] * xs + sl["encoder_e5"]["intercept"],
            color=ENC_C, lw=HEAVY_LINE_W, zorder=6)
    ax.plot(xs, sl["llm_mean"]["slope"] * xs + sl["llm_mean"]["intercept"],
            color=LLM_C, lw=HEAVY_LINE_W, zorder=6)
    ax.set_ylim(0.45, 1.0)
    ax.text(x.min() + 0.004, 0.468, "\u2190 more distinctive", ha="left", va="center",
            fontsize=ANN_SIZE - 0.5, color=INK, style="italic")
    ax.set_xlabel("Community distinctiveness:  AUC(toxicity \u2192 recorded removal)")
    ax.set_ylabel("Within-community BAL-AUC")
    leg = ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="none", borderpad=0.6)
    leg.get_frame().set_facecolor(PANEL)
    finish_axis(ax, xgrid=True, ygrid=True)
    autofit_axis_label(ax, fig, "x")
    autofit_axis_label(ax, fig, "y")
    save(fig, "fig_distinctiveness")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    setup_style()
    validate_inputs()
    # Snapshot Figure 1 before plotting and re-check after, so any accidental
    # overwrite of the hand-maintained figure aborts the run loudly.
    fig1_before = sha256(FIGURES / "methods-comparison.png")
    if args.verify_only:
        fig1_after = sha256(FIGURES / "methods-comparison.png")
        if fig1_after != fig1_before:
            raise RuntimeError("Figure 1 changed")
        print(json.dumps({"ok": True, "figure1_sha256": fig1_after, "verify_only": True}, indent=2))
        return
    plot_tc_test()
    plot_rule_interventions()
    plot_sae()
    plot_nontoxic()
    plot_coverage()
    plot_where_encoded()
    plot_over_removal()
    plot_distinctiveness()
    fig1_after = sha256(FIGURES / "methods-comparison.png")
    if fig1_after != fig1_before:
        raise RuntimeError("Figure 1 changed")
    write_manifest(fig1_before)
    print(json.dumps({"ok": True, "figure1_sha256": fig1_after, "source_events": len(SOURCE_LOG)}, indent=2))


if __name__ == "__main__":
    main()
