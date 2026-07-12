"""Figures for the CAP-threshold sensitivity analysis. Each function takes the
DataFrames from fibrosis_lib and a `dark` flag, and returns a matplotlib Figure.
Used inline by the notebook (light) and by the artifact builder (light + dark)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import fibrosis_lib as fl


def _S(dark):
    if dark:
        return dict(surface="#1a1a19", ink="#f2f3f1", sec="#c3c2b7", mut="#898781",
                    grid="#2c2c2a", base="#383835", accent="#3987e5", nomasld="#4a4c48",
                    ramp=["#12467f", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"],
                    neg="#3987e5", pos="#e34948")
    return dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", mut="#898781",
                grid="#e1e0d9", base="#c3c2b7", accent="#2a78d6", nomasld="#d6d7d2",
                ramp=["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
                neg="#2a78d6", pos="#e34948")


def _frame(ax, S):
    ax.set_facecolor(S["surface"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(S["base"]); ax.spines[sp].set_linewidth(1)
    ax.tick_params(colors=S["mut"], labelsize=9.5)


def _short_cols(df):
    """Map exposure/cutoff column labels to clean two-line tick labels."""
    labmap = {fl.collabel(cap, src, lvl): fl.shortlabel(cap, src, lvl)
              for cap, src, lvl in fl.CUTOFFS}
    out = []
    for c in df.columns:
        if c in labmap:
            out.append(labmap[c])
        elif c.startswith("LSM"):
            out.append("LSM\n(cont.)")
        elif c.startswith("CAP (cont"):
            out.append("CAP\n(cont.)")
        else:
            out.append(c)
    return out


def prevalence_stacked(prev, dark=False):
    S = _S(dark)
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=170)
    fig.patch.set_facecolor(S["surface"]); _frame(ax, S)
    cols = list(prev.columns)
    x = np.arange(len(cols))
    segs = [("No MASLD (no steatosis)", S["nomasld"], "No MASLD")] + [
        (f"MASLD F{s}", S["ramp"][s], f"F{s}") for s in range(5)]
    bottom = np.zeros(len(cols))
    for row, color, lab in segs:
        vals = prev.loc[row].values.astype(float)
        ax.bar(x, vals, 0.74, bottom=bottom, color=color, label=lab,
               edgecolor=S["surface"], linewidth=1.2, zorder=3)
        bottom += vals
    # label the No MASLD share on top of its segment
    for xi, v in zip(x, prev.loc["No MASLD (no steatosis)"].values.astype(float)):
        if v > 3:
            ax.text(xi, v / 2, f"{v:.0f}", ha="center", va="center",
                    fontsize=9, color=S["ink"], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(_short_cols(prev), fontsize=8.5, color=S["ink"])
    ax.set_ylabel("Weighted prevalence (%)", color=S["sec"], fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_title("Population composition by CAP steatosis cutoff",
                 color=S["ink"], fontsize=13, fontweight="bold", loc="left", pad=10)
    handles = [Patch(color=c, label=l) for _, c, l in segs]
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=6,
              loc="lower center", bbox_to_anchor=(0.5, -0.28), labelcolor=S["ink"])
    fig.tight_layout()
    return fig


def f2plus_sensitivity(sens, dark=False):
    S = _S(dark)
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=170)
    fig.patch.set_facecolor(S["surface"]); _frame(ax, S)
    ax.grid(axis="y", color=S["grid"], lw=0.8, zorder=0)
    x = np.arange(len(sens))
    ax.errorbar(x, sens["f2plus"], yerr=sens["hw"], fmt="o-", color=S["accent"],
                capsize=4, lw=2, markersize=8, markerfacecolor=S["accent"],
                markeredgecolor=S["surface"], markeredgewidth=1.2,
                ecolor=S["sec"], zorder=3)
    for xi, v in zip(x, sens["f2plus"]):
        ax.text(xi, v + max(sens["hw"]) + 0.5, f"{v:.1f}", ha="center",
                fontsize=9, color=S["sec"], fontweight="bold")
    # mark the 288 baseline
    base_i = sens.index[sens["cap"] == 288]
    if len(base_i):
        bi = list(sens["cap"]).index(288)
        ax.axvline(bi, color=S["base"], lw=1, ls=":", zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(sens["label"], fontsize=8.5, color=S["ink"])
    ax.set_ylabel("MASLD significant fibrosis, F2+ (%)", color=S["sec"], fontsize=10.5)
    ax.set_ylim(0, max(sens["f2plus"]) + 2.2)
    ax.set_title("Significant-fibrosis prevalence is sensitive to the CAP cutoff",
                 color=S["ink"], fontsize=13, fontweight="bold", loc="left", pad=10)
    fig.tight_layout()
    return fig


SEX_COLORS = {"Male": {"l": "#2a78d6", "d": "#3987e5"},
              "Female": {"l": "#eb6834", "d": "#d95926"}}

STAGE_CATS = ["<F1", "F1", "F2–3", ">F3"]


def _stage_ramp(dark):
    # <F1 neutral, then blue by severity
    if dark:
        return {"<F1": "#4a4c48", "F1": "#256abf", "F2–3": "#5598e7", ">F3": "#b7d3f6"}
    return {"<F1": "#d6d7d2", "F1": "#9ec5f4", "F2–3": "#2a78d6", ">F3": "#0d366b"}


def _strat_order(df, col):
    bands = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-80"]
    return [f"{s} {a}" for s in ("Female", "Male") for a in bands]


def stage_distribution_fig(dist, dark=False):
    """100% stacked bars of the LSM stage groups (<F1/F1/F2–3/>F3), one per sex×age stratum."""
    S = _S(dark); ramp = _stage_ramp(dark)
    dist = dist.copy()
    dist["stratum"] = dist["sex"] + " " + dist["age"]
    order = _strat_order(dist, "stratum")
    piv = dist.pivot(index="stratum", columns="category", values="pct").reindex(order)[STAGE_CATS]
    fig, ax = plt.subplots(figsize=(12.4, 5.6), dpi=170)
    fig.patch.set_facecolor(S["surface"]); _frame(ax, S); ax.grid(axis="x", visible=False)
    x = np.arange(len(order)); bottom = np.zeros(len(order))
    for cat in STAGE_CATS:
        vals = piv[cat].values.astype(float)
        ax.bar(x, vals, 0.74, bottom=bottom, color=ramp[cat], label=cat,
               edgecolor=S["surface"], linewidth=1.1, zorder=3)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 4:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if cat in ("F2–3", ">F3") else S["ink"],
                        fontweight="bold")
        bottom += vals
    ax.axvline(5.5, color=S["base"], lw=1, ls=":")  # Female | Male divider
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace(" ", "\n") for o in order], fontsize=8.3, color=S["ink"])
    ax.set_ylabel("Weighted prevalence (%)", color=S["sec"], fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_title("LSM fibrosis-stage distribution by sex × age band",
                 color=S["ink"], fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.26), labelcolor=S["ink"], title=None)
    fig.tight_layout()
    return fig


def stage_trend_fig(dist, dark=False):
    """Trend of the two clinically important groups (F2–3, >F3) by age; Male/Female with 95% CIs."""
    S = _S(dark)
    bands = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-80"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), dpi=170)
    fig.patch.set_facecolor(S["surface"])
    for ax, cat in zip(axes, ["F2–3", ">F3"]):
        _frame(ax, S); ax.grid(axis="y", color=S["grid"], lw=0.8, zorder=0)
        x = np.arange(len(bands))
        for sex in ("Male", "Female"):
            sub = (dist[(dist["category"] == cat) & (dist["sex"] == sex)]
                   .set_index("age").reindex(bands))
            col = SEX_COLORS[sex]["d" if dark else "l"]
            ax.errorbar(x + (0.06 if sex == "Male" else -0.06), sub["pct"],
                        yerr=(sub["pct"] - sub["ci_lo"]).clip(lower=0), fmt="o-",
                        color=col, capsize=3, lw=2, markersize=6.5,
                        markeredgecolor=S["surface"], markeredgewidth=1, ecolor=col,
                        label=sex, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=9, color=S["ink"])
        ax.set_ylim(0, None)
        ax.set_xlabel("Age band (years)", color=S["sec"], fontsize=9.5)
        ax.set_title(cat, color=S["ink"], fontsize=12, fontweight="bold", loc="left", pad=6)
    axes[0].set_ylabel("Weighted prevalence (%)", color=S["sec"], fontsize=10.5)
    axes[1].text(0.5, 0.97, ">F3 (F4): sparse cells — wide CIs", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=8.5, color=S["mut"], style="italic")
    axes[0].legend(frameon=False, fontsize=10, loc="upper left", labelcolor=S["ink"])
    fig.suptitle("Significant-to-advanced fibrosis by sex and age (LSM)",
                 color=S["ink"], fontsize=13, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    return fig


def stratified_prevalence_fig(sp, dark=False):
    """3 panels (steatosis, F2+, F3+); x=age band; Male/Female lines with 95% CIs."""
    S = _S(dark)
    outcomes = ["Steatosis (MASLD)", "MASLD F2+ (significant)", "MASLD F3+ (advanced)"]
    bands = sorted(sp["age"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), dpi=170)
    fig.patch.set_facecolor(S["surface"])
    for ax, oc in zip(axes, outcomes):
        _frame(ax, S); ax.grid(axis="y", color=S["grid"], lw=0.8, zorder=0)
        x = np.arange(len(bands))
        for sex in ("Male", "Female"):
            sub = sp[(sp["outcome"] == oc) & (sp["sex"] == sex)].set_index("age").reindex(bands)
            col = SEX_COLORS[sex]["d" if dark else "l"]
            ax.errorbar(x + (0.06 if sex == "Male" else -0.06), sub["pct"],
                        yerr=(sub["pct"] - sub["ci_lo"]).clip(lower=0), fmt="o-",
                        color=col, capsize=3, lw=2, markersize=6.5,
                        markeredgecolor=S["surface"], markeredgewidth=1,
                        ecolor=col, label=sex, zorder=3, alpha=0.95)
        ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=9, color=S["ink"], rotation=0)
        ax.set_ylim(0, None)
        ax.set_title(oc, color=S["ink"], fontsize=11.5, fontweight="bold", loc="left", pad=7)
        ax.set_xlabel("Age band (years)", color=S["sec"], fontsize=9.5)
    axes[0].set_ylabel("Weighted prevalence (%)", color=S["sec"], fontsize=10.5)
    axes[2].text(0.5, 0.97, "sparse cells — wide CIs", transform=axes[2].transAxes,
                 ha="center", va="top", fontsize=8.5, color=S["mut"], style="italic")
    axes[0].legend(frameon=False, fontsize=10, loc="upper left", labelcolor=S["ink"])
    fig.suptitle("MASLD steatosis & fibrosis prevalence by sex and age (CAP≥288)",
                 color=S["ink"], fontsize=13, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    return fig


def stratified_corr_heatmap(cdf, dark=False):
    """Heatmap: rows = sex x age strata, cols = full-sample risks; MASLD F-stage rho."""
    S = _S(dark)
    cmap = LinearSegmentedColormap.from_list("div", [S["neg"], S["surface"], S["pos"]])
    fig, ax = plt.subplots(figsize=(10.6, 6.4), dpi=170)
    fig.patch.set_facecolor(S["surface"])
    M = cdf.values.astype(float)
    im = ax.imshow(M, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(np.arange(cdf.shape[1])); ax.set_xticklabels(cdf.columns, fontsize=9.5,
                                                               color=S["ink"], rotation=30, ha="right")
    ax.set_yticks(np.arange(cdf.shape[0])); ax.set_yticklabels(cdf.index, fontsize=9.5, color=S["ink"])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # divider between Female and Male blocks
    fem = sum(str(i).startswith("Female") for i in cdf.index)
    ax.axhline(fem - 0.5, color=S["surface"], lw=3)
    for i in range(cdf.shape[0]):
        for j in range(cdf.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.4 else S["ink"])
    ax.set_title("MASLD F-stage (CAP≥288) ρ with metabolic risk, by sex × age",
                 color=S["ink"], fontsize=12.5, fontweight="bold", loc="left", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=S["mut"], labelsize=8)
    cb.set_label("Spearman ρ", color=S["sec"], fontsize=9)
    fig.tight_layout()
    return fig


def correlation_heatmap(corr, dark=False):
    S = _S(dark)
    cmap = LinearSegmentedColormap.from_list("div", [S["neg"], S["surface"], S["pos"]])
    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=170)
    fig.patch.set_facecolor(S["surface"])
    M = corr.values.astype(float)
    vmax = 0.7
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_xticklabels(_short_cols(corr), fontsize=8, color=S["ink"], rotation=0)
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_yticklabels(corr.index, fontsize=10, color=S["ink"])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = M[i, j]
            tc = "#ffffff" if abs(v) > 0.42 else S["ink"]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6, color=tc)
    # separate the two continuous reference columns from the categorical block
    ax.axvline(1.5, color=S["surface"], lw=3)
    ax.set_title("Weighted Spearman ρ of each exposure with metabolic risk",
                 color=S["ink"], fontsize=13, fontweight="bold", loc="left", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=S["mut"], labelsize=8)
    cb.set_label("Spearman ρ", color=S["sec"], fontsize=9)
    fig.tight_layout()
    return fig
