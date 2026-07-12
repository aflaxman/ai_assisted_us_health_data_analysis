"""Driver: compute the CAP-cutoff prevalence + correlation tables, write CSVs to
outputs/, and render the sensitivity figures (light + dark) to a target dir.

    python -P 02_cutoff_sensitivity.py <fig_out_dir>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fibrosis_lib as fl
import fibrosis_plots as fp

FIGDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

coh = fl.build_cohort()
prev = fl.prevalence_table(coh)
corr = fl.correlation_table(coh)
sens = fl.f2plus_sensitivity(coh)

prev.to_csv(OUT / "prevalence_by_cap_cutoff.csv")
corr.to_csv(OUT / "correlations_by_cap_cutoff.csv")
print(f"n={len(coh)}  wrote CSVs to {OUT}")

for dark in (False, True):
    tag = "dark" if dark else "light"
    fp.prevalence_stacked(prev, dark).savefig(
        FIGDIR / f"fig3_prev_stacked_{tag}.png", facecolor="#1a1a19" if dark else "#fcfcfb", bbox_inches="tight")
    fp.f2plus_sensitivity(sens, dark).savefig(
        FIGDIR / f"fig4_f2plus_{tag}.png", facecolor="#1a1a19" if dark else "#fcfcfb", bbox_inches="tight")
    fp.correlation_heatmap(corr, dark).savefig(
        FIGDIR / f"fig5_corr_heatmap_{tag}.png", facecolor="#1a1a19" if dark else "#fcfcfb", bbox_inches="tight")
print(f"wrote figures to {FIGDIR}")
