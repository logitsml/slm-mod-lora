"""Render figures/fig_crossmodel.pdf from the cross-model and Gemma SAE result files.

Panel (a) necessity flip-to-keep against the number of ablated toxicity features, one line per model.
Panel (b) sufficiency induction for toxicity versus the strongest matched control, one group per model.
  run: python -m pipeline.kumar_mod.make_crossmodel_figure
"""
import os, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
SAE = ROOT / "results" / "kumar_mod" / "sae"
KM = ROOT / "results" / "kumar_mod"

def jload(p):
    return json.load(open(p))

# Llama/Qwen come from the cross-model replication; Gemma necessity/sufficiency
# live in the primary single-model result files (no K-sweep there, just K=10).
llama = jload(SAE / "crossmodel_llama.json")
qwen = jload(SAE / "crossmodel_qwen.json")
gnec = jload(KM / "sae_causal_dosedense.json")
gsuf = jload(KM / "sae_sufficiency.json")

C = {"Gemma-3-12B": "#444444", "Llama-3.1-8B": "#1f77b4", "Qwen2.5-7B": "#d62728"}
Ks = [10, 20, 40, 80]
# Flip-to-keep rate as a function of how many top toxicity features are ablated.
# K-sweep keys are JSON strings, hence str(k).
nec = {"Llama-3.1-8B": [llama["necessity"]["necessity_ksweep_flip"][str(k)] for k in Ks],
       "Qwen2.5-7B": [qwen["necessity"]["necessity_ksweep_flip"][str(k)] for k in Ks]}
# Gemma is a single point: its 16k SAE already saturates necessity at K=10,
# so it is plotted as a star rather than a sweep line.
gemma_k10 = gnec["flip_rate"]["all"]["tox"]

def tox_and_strongest_control(induction):
    # Compare the toxicity feature against the *worst-case* control: the strongest
    # non-toxicity control arm (each control clamps a whole matched feature set to
    # its own p95; we take the arm with the highest induction). If even that one
    # barely moves benign comments, the effect is toxicity-specific.
    tox = induction["tox_p95"]
    ctrl = max(v for k, v in induction.items() if k != "tox_p95")
    return tox, ctrl

suff = {"Gemma-3-12B": tox_and_strongest_control(gsuf["induction_rate"]),
        "Llama-3.1-8B": tox_and_strongest_control(llama["sufficiency"]["induction_rate"]),
        "Qwen2.5-7B": tox_and_strongest_control(qwen["sufficiency"]["induction_rate"])}

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

# Even x spacing for the K values; Ks are the tick labels, not the positions,
# so K=10/20/40/80 render equidistant rather than on a log/linear scale.
x = np.arange(len(Ks))
for m, ys in nec.items():
    ax1.plot(x, ys, marker="o", ms=5, lw=2, color=C[m], label=m)
# Gemma's single K=10 point sits at the leftmost tick; the dotted line carries
# its value across so it can be read against the Llama/Qwen sweeps at any K.
ax1.plot([0], [gemma_k10], marker="*", ms=14, color=C["Gemma-3-12B"], label="Gemma-3-12B")
ax1.axhline(gemma_k10, xmin=0.0, xmax=1.0, color=C["Gemma-3-12B"], lw=1, ls=":", alpha=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(Ks)
ax1.set_ylim(-0.03, 1.06); ax1.set_xlabel("Toxicity features ablated ($K$)")
ax1.set_ylabel("Removals flipped to keep")
ax1.set_title("(a) Necessity scales with coverage", fontsize=9, loc="left")
ax1.annotate("Gemma saturates at $K{=}10$\n(16k SAE)", xy=(0, gemma_k10), xytext=(0.55, 0.62),
             fontsize=7.2, color=C["Gemma-3-12B"], ha="left",
             arrowprops=dict(arrowstyle="->", color=C["Gemma-3-12B"], lw=0.8))
ax1.legend(frameon=False, fontsize=7.5, loc="center right")

models = list(suff.keys()); xb = np.arange(len(models)); w = 0.36
tox = [suff[m][0] for m in models]; ctl = [suff[m][1] for m in models]
# Paired bars per model: toxicity feature (in the model's colour) vs strongest
# control (grey). The colour-coded tox bar keeps the legend consistent with (a).
ax2.bar(xb - w / 2, tox, w, color=[C[m] for m in models], label="toxicity")
ax2.bar(xb + w / 2, ctl, w, color="#bbbbbb", label="strongest control")
for i, (t, c) in enumerate(zip(tox, ctl)):
    ax2.text(i - w / 2, t + 0.02, f"{t:.2f}", ha="center", fontsize=7)
    # Controls below 0.01 print "≈0" so a near-zero bar
    # is not mislabelled as an exact zero.
    ax2.text(i + w / 2, c + 0.02, f"{c:.2f}" if c >= 0.01 else "$\\approx$0", ha="center", fontsize=7)
ax2.set_xticks(xb); ax2.set_xticklabels(["Gemma\n3-12B", "Llama\n3.1-8B", "Qwen\n2.5-7B"], fontsize=7.5)
ax2.set_ylim(0, 1.0); ax2.set_ylabel("Benign comments induced to remove")
ax2.set_title("(b) Sufficiency is toxicity-specific", fontsize=9, loc="left")
ax2.legend(frameon=False, fontsize=7.5, loc="upper right")

fig.tight_layout(w_pad=2.0)
out = ROOT / "figures" / "fig_crossmodel.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

# Register this figure in the shared manifest so every shipped figure has
# recorded input provenance (same schema make_paper_figures.py emits).
import hashlib, json as _json
man_path = ROOT / "figures" / "figure_sources.json"
if man_path.exists():
    man = _json.loads(man_path.read_text())
    def _sha(rel):
        return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    ins = [{"file": f"results/kumar_mod/sae/crossmodel_{m}.json",
            "field": "necessity/sufficiency/cert blocks",
            "sha256": _sha(f"results/kumar_mod/sae/crossmodel_{m}.json")} for m in ("llama", "qwen")]
    man["sources"] = [s for s in man.get("sources", []) if "crossmodel_" not in s.get("file", "")] + ins
    outs = man.get("outputs", [])
    if "figures/fig_crossmodel.pdf" not in outs:
        man["outputs"] = outs + ["figures/fig_crossmodel.pdf"]
    man_path.write_text(_json.dumps(man, indent=1))
    print("manifest updated with fig_crossmodel sources")
