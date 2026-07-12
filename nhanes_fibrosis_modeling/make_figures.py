"""Render prevalence + correlation figures comparing LSM-alone vs MASLD (CAP>=288)
fibrosis definitions in NHANES 2017-March 2020 pre-pandemic. Reads results.json
(produced by compute step) and writes light+dark PNGs to the given output dir."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SP = sys.argv[1] if len(sys.argv) > 1 else "."
R = json.load(open(f"{SP}/results.json"))

prevA, prevB = R["prev"]["A"], R["prev"]["B"]
corr = R["corr"]
stages = ["F0", "F1", "F2", "F3", "F4"]


def style(dark):
    if dark:
        return dict(surface="#1a1a19", ink="#ffffff", sec="#c3c2b7", mut="#898781",
                    grid="#2c2c2a", base="#383835",
                    blue="#3987e5", orange="#d95926", aqua="#199e70")
    return dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", mut="#898781",
                grid="#e1e0d9", base="#c3c2b7",
                blue="#2a78d6", orange="#eb6834", aqua="#1baf7a")


def apply(ax, S):
    ax.set_facecolor(S["surface"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(S["base"]); ax.spines[sp].set_linewidth(1)
    ax.tick_params(colors=S["mut"], labelsize=10)
    ax.grid(color=S["grid"], lw=0.8, zorder=0)


def fig1(dark):
    S = style(dark)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=170)
    fig.patch.set_facecolor(S["surface"])
    x = np.arange(5); w = 0.38

    a = axes[0]; apply(a, S); a.grid(axis="x", visible=False)
    va = [prevA[s][0] for s in stages]; ea = [prevA[s][1] for s in stages]
    vb = [prevB[s][0] for s in stages]; eb = [prevB[s][1] for s in stages]
    a.bar(x - w/2, va, w, yerr=ea, capsize=3, color=S["blue"], label="LSM alone",
          zorder=3, error_kw=dict(ecolor=S["sec"], lw=1.1))
    a.bar(x + w/2, vb, w, yerr=eb, capsize=3, color=S["orange"],
          label="MASLD (LSM + CAP≥288)", zorder=3, error_kw=dict(ecolor=S["sec"], lw=1.1))
    for xi, v in zip(x - w/2, va):
        a.text(xi, v + 1.4, f"{v:.1f}", ha="center", fontsize=8.5, color=S["sec"], fontweight="bold")
    for xi, v in zip(x + w/2, vb):
        a.text(xi, v + 1.4, f"{v:.1f}", ha="center", fontsize=8.5, color=S["sec"], fontweight="bold")
    a.set_xticks(x); a.set_xticklabels(stages, color=S["ink"], fontsize=11)
    a.set_ylabel("Weighted prevalence (%)", color=S["sec"], fontsize=10.5)
    a.set_ylim(0, 94)
    a.set_title("Prevalence by fibrosis stage", color=S["ink"], fontsize=12.5,
                fontweight="bold", loc="left", pad=8)
    a.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=S["ink"])

    b = axes[1]; apply(b, S); b.grid(axis="x", visible=False)
    labs = ["Any\n(F1+)", "Significant\n(F2+)", "Advanced\n(F3+)", "Cirrhosis-\nrange (F4)"]
    cum = lambda pr, k: sum(pr[f"F{s}"][0] for s in range(k, 5))
    ca = [cum(prevA, k) for k in (1, 2, 3, 4)]
    cb = [cum(prevB, k) for k in (1, 2, 3, 4)]
    xc = np.arange(4)
    b.bar(xc - w/2, ca, w, color=S["blue"], zorder=3, label="LSM alone")
    b.bar(xc + w/2, cb, w, color=S["orange"], zorder=3, label="MASLD (LSM + CAP≥288)")
    for xi, v in zip(xc - w/2, ca):
        b.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, color=S["sec"], fontweight="bold")
    for xi, v in zip(xc + w/2, cb):
        b.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, color=S["sec"], fontweight="bold")
    b.set_xticks(xc); b.set_xticklabels(labs, color=S["ink"], fontsize=10)
    b.set_ylabel("Weighted prevalence (%)", color=S["sec"], fontsize=10.5)
    b.set_ylim(0, 30)
    b.set_title("Prevalence above clinical thresholds", color=S["ink"], fontsize=12.5,
                fontweight="bold", loc="left", pad=8)
    b.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=S["ink"])
    fig.tight_layout()
    out = f"{SP}/fig1_prev_{'dark' if dark else 'light'}.png"
    fig.savefig(out, facecolor=S["surface"], bbox_inches="tight"); plt.close(fig)
    return out


def fig2(dark):
    S = style(dark)
    order = sorted(corr.keys(), key=lambda k: abs(corr[k]["stageB"][0]), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), dpi=170, sharey=True)
    fig.patch.set_facecolor(S["surface"])
    y = np.arange(len(order))[::-1]; h = 0.38

    a = axes[0]; apply(a, S); a.grid(axis="y", visible=False)
    for yi, k in zip(y, order):
        rA, rB = corr[k]["stageA"], corr[k]["stageB"]
        a.barh(yi + h/2, rA[0], h, color=S["blue"], zorder=3,
               xerr=[[rA[0]-rA[1]], [rA[2]-rA[0]]], error_kw=dict(ecolor=S["sec"], lw=1))
        a.barh(yi - h/2, rB[0], h, color=S["orange"], zorder=3,
               xerr=[[rB[0]-rB[1]], [rB[2]-rB[0]]], error_kw=dict(ecolor=S["sec"], lw=1))
    a.axvline(0, color=S["base"], lw=1)
    a.set_yticks(y); a.set_yticklabels(order, color=S["ink"], fontsize=10.5)
    a.set_xlabel("Weighted Spearman ρ with metabolic risk", color=S["sec"], fontsize=10.5)
    a.set_title("Categorical F-stage", color=S["ink"], fontsize=12.5, fontweight="bold", loc="left", pad=8)
    a.set_xlim(-0.45, 0.74)
    a.legend(handles=[Patch(color=S["blue"], label="F-stage: LSM alone"),
                      Patch(color=S["orange"], label="F-stage: MASLD (CAP≥288)")],
             frameon=False, fontsize=9, loc="lower right", labelcolor=S["ink"])

    b = axes[1]; apply(b, S); b.grid(axis="y", visible=False)
    for yi, k in zip(y, order):
        rL, rC = corr[k]["LSM"], corr[k]["CAP"]
        b.barh(yi + h/2, rL[0], h, color=S["blue"], zorder=3,
               xerr=[[rL[0]-rL[1]], [rL[2]-rL[0]]], error_kw=dict(ecolor=S["sec"], lw=1))
        b.barh(yi - h/2, rC[0], h, color=S["aqua"], zorder=3,
               xerr=[[rC[0]-rC[1]], [rC[2]-rC[0]]], error_kw=dict(ecolor=S["sec"], lw=1))
    b.axvline(0, color=S["base"], lw=1)
    b.set_xlabel("Weighted Spearman ρ with metabolic risk", color=S["sec"], fontsize=10.5)
    b.set_title("Continuous exposures", color=S["ink"], fontsize=12.5, fontweight="bold", loc="left", pad=8)
    b.set_xlim(-0.45, 0.74)
    b.legend(handles=[Patch(color=S["blue"], label="LSM (stiffness)"),
                      Patch(color=S["aqua"], label="CAP (steatosis)")],
             frameon=False, fontsize=9, loc="lower right", labelcolor=S["ink"])
    fig.tight_layout()
    out = f"{SP}/fig2_corr_{'dark' if dark else 'light'}.png"
    fig.savefig(out, facecolor=S["surface"], bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    for d in (False, True):
        print(fig1(d)); print(fig2(d))
    print("done")
