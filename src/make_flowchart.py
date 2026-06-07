#!/usr/bin/env python3
"""Professional pipeline flowchart for the paper (vector SVG + high-DPI PNG)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family": "DejaVu Sans"})

C = dict(data="#27496d", prep="#0b6e6e", geom="#2e7d32", bayes="#1f4e79",
         valid="#b7791f", ml="#6b46c1", res="#7a1f3d", edge="#3a3a3a")
fig, ax = plt.subplots(figsize=(11.5, 14.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, text, color, fs=10.5, tc="white", bold=True, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.3,rounding_size=1.1",
                 linewidth=1.3, edgecolor=C["edge"], facecolor=color, alpha=alpha, mutation_scale=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25, zorder=5)
    return dict(x=x, y=y, w=w, h=h)
def arrow(a, b, style="-|>", lw=1.7, color=None, rad=0.0, sa="bottom", ea="top"):
    p = {"top": (a["x"], a["y"]+a["h"]/2), "bottom": (a["x"], a["y"]-a["h"]/2),
         "left": (a["x"]-a["w"]/2, a["y"]), "right": (a["x"]+a["w"]/2, a["y"])}
    q = {"top": (b["x"], b["y"]+b["h"]/2), "bottom": (b["x"], b["y"]-b["h"]/2),
         "left": (b["x"]-b["w"]/2, b["y"]), "right": (b["x"]+b["w"]/2, b["y"])}
    ax.add_patch(FancyArrowPatch(p[sa], q[ea], arrowstyle=style, mutation_scale=15,
                 lw=lw, color=color or C["edge"], connectionstyle=f"arc3,rad={rad}", zorder=1))
def lane(y0, y1, label, color):
    ax.add_patch(FancyBboxPatch((4.2, y1), 94.6, y0-y1, boxstyle="round,pad=0.2,rounding_size=0.6",
                 linewidth=0, facecolor=color, alpha=0.06, zorder=0))
    ax.text(2.1, 0.5*(y0+y1), label, ha="center", va="center", fontsize=8.2, color=color,
            fontweight="bold", alpha=0.9, rotation=90)

# Title
ax.text(50, 98.2, "Path-length exponent of parton energy loss across collision systems",
        ha="center", fontsize=15, fontweight="bold", color="#1a1a1a")
ax.text(50, 95.6, "Bayesian extraction + probabilistic-ML / simulation-based inference  —  end-to-end pipeline",
        ha="center", fontsize=10.5, color="#555")

# ---- DATA ----
lane(94, 84.5, "DATA", C["data"])
d1 = box(20, 89.5, 30, 5.4, "CMS-HIN-25-014  (ins3123773)\nO+O, Ne+Ne, Pb+Pb, p+Pb", C["data"], 9.2)
d2 = box(50, 89.5, 23, 5.4, "Xe+Xe\n(ins1692558)", C["data"], 9.2)
d3 = box(80, 89.5, 30, 5.4, "Pb+Pb (ins1496050)\npp reference + centrality", C["data"], 9.2)

# ---- PREPROCESS ----
lane(83.5, 73, "PREPROCESSING", C["prep"])
p0 = box(50, 80.6, 40, 4.6, "4 systems:  charged-particle $R_{AA}(p_T)$,  $p_T \\geq 8$ GeV", C["prep"], 10)
p1 = box(22, 75.2, 28, 4.8, "Correlation-length\ncovariance ($\\xi=4$ bins)", C["prep"], 9.2)
p2 = box(52, 75.2, 26, 4.8, "Local index $a(p_T)$\nfrom pp spectrum", C["prep"], 9.2)
p3 = box(80, 75.2, 26, 4.8, "$\\sqrt{s}$ correction\n($x_T$-scaling + $f_s$)", C["prep"], 9.2)

# ---- GEOMETRY ----
lane(72, 61.5, "GEOMETRY", C["geom"])
g1 = box(28, 68, 30, 4.8, "Optical Glauber\n$\\langle L\\rangle,\\langle N_{part}\\rangle,$ area, $\\rho$", C["geom"], 9.4)
g2 = box(72, 68, 32, 4.8, "Monte-Carlo Glauber\n(nucleon sampling, no ROOT)", C["geom"], 9.4)
gx = box(50, 63.2, 46, 4.2, "cross-check: $\\langle N_{part}\\rangle^{1/3}$ ratios agree to $\\pm0.04$", C["geom"], 9.6, alpha=0.92)

# ---- BAYESIAN ----
lane(60.5, 49, "BAYESIAN", C["bayes"])
b0 = box(50, 57, 44, 4.8, "Forward model $\\Delta p_T=\\kappa\\,\\rho\\,G^{n}p_T^{\\beta}$  +  correlated likelihood (emcee MCMC)", C["bayes"], 9.6)
b1 = box(28, 51.4, 30, 4.6, "effective  $n \\approx 1.8$\n(geometry $\\times$ density)", C["bayes"], 9.6)
b2 = box(72, 51.4, 32, 4.6, "pure  $n \\approx 0.5$\n(density-normalized)", C["bayes"], 9.6)

# ---- VALIDATION / SELECTION ----
lane(48, 33.5, "VALIDATION", C["valid"])
v1 = box(16, 43.4, 21, 5.6, "Bayes factors\n$n{=}1/2/3$\n$\\Rightarrow$ radiative", C["valid"], 8.8)
v2 = box(39, 43.4, 21, 5.6, "Coverage test\n0.70 / 0.87\n(calibrated)", C["valid"], 8.8)
v3 = box(62, 43.4, 21, 5.6, "Universality\nO+O$\\to$Pb+Pb\n(no break)", C["valid"], 8.8)
v4 = box(85, 43.4, 22, 5.6, "Systematic\nbudget ($\\sqrt{s}$,\nbaseline, Glauber)", C["valid"], 8.8)
vsum = box(50, 36.2, 64, 4.0, "$n = 1.85 \\pm 0.15\\,(\\mathrm{stat}) \\pm 0.05\\,(\\mathrm{syst})$   —  radiative-dominated, universal", C["valid"], 9.8, alpha=0.95)

# ---- ML PILLAR ----
lane(32.5, 16, "MACHINE LEARNING", C["ml"])
m1 = box(27, 27.4, 32, 5.4, "Probabilistic emulator (GP)\n+ leave-one-system-out\n(predict held-out Ne+Ne)", C["ml"], 8.8)
m2 = box(72, 27.4, 34, 5.4, "Neural Posterior Estimation\n(normalizing flow) + SBC\namortized inference", C["ml"], 8.8)
mx = box(50, 19.4, 56, 4.2, "ML posterior  $n = 1.78$  $\\approx$  MCMC  $n=1.82$   $\\checkmark$  (calibrated)", C["ml"], 9.6, alpha=0.95)

# ---- RESULTS ----
lane(15, 4.5, "RESULT", C["res"])
r0 = box(50, 9.6, 80, 6.6,
         "Radiative-dominated energy loss ($n{=}2$ favoured; $n{=}1,3$ rejected)\n"
         "Universal scaling O+O $\\to$ Pb+Pb  •  effective $n=1.85\\pm0.15$,  pure $\\approx 0.5$ (consistent with prior)\n"
         "Cross-validated by Glauber, coverage, and calibrated neural posterior estimation",
         C["res"], 9.6)

# ---- arrows ----
for d in (d1, d2, d3): arrow(d, p0)
arrow(p0, p1, rad=0.0, sa="bottom", ea="top"); arrow(p0, p2); arrow(p0, p3)
for p in (p1, p2, p3): arrow(p, g1 if p is p1 else (gx if p is p2 else g2), color=C["edge"])
arrow(g1, gx); arrow(g2, gx); arrow(gx, b0)
arrow(b0, b1); arrow(b0, b2)
arrow(b1, v1, color=C["valid"]); arrow(b1, v2, color=C["valid"]); arrow(b2, v3, color=C["valid"]); arrow(b2, v4, color=C["valid"])
for v in (v1, v2, v3, v4): arrow(v, vsum, color=C["valid"])
arrow(vsum, m1, color=C["ml"]); arrow(vsum, m2, color=C["ml"])
arrow(m1, mx, color=C["ml"]); arrow(m2, mx, color=C["ml"])
arrow(mx, r0, color=C["res"])
# feedback dashed: ML validates MCMC
ax.add_patch(FancyArrowPatch((m2["x"]+m2["w"]/2, m2["y"]), (b2["x"]+b2["w"]/2, b2["y"]),
             arrowstyle="-|>", mutation_scale=12, lw=1.3, color=C["ml"], ls=(0, (4, 3)),
             connectionstyle="arc3,rad=-0.32", zorder=1))
ax.text(95.5, 39.5, "validates", fontsize=8, color=C["ml"], rotation=90, ha="center", va="center", style="italic")

fig.savefig("/tmp/out/pipeline_flowchart.svg", bbox_inches="tight")
fig.savefig("/tmp/out/pipeline_flowchart.png", dpi=200, bbox_inches="tight")
print("saved pipeline_flowchart.svg + .png")
