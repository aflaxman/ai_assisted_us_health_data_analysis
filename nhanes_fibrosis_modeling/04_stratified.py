"""Driver for the sex x age-band stratified analysis: write CSVs to outputs/ and
render figures (light + dark) to a target dir.

    python -P 04_stratified.py <fig_out_dir>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fibrosis_lib as fl
import fibrosis_strat as fs
import fibrosis_plots as fp

FIGDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)

coh = fl.build_cohort()
counts = fs.cell_counts(coh)
prev = fs.stratified_prevalence(coh)
corr = fs.stratified_correlation(coh)

counts.to_csv(OUT / "stratified_cell_counts.csv", index=False)
prev.to_csv(OUT / "stratified_prevalence.csv", index=False)
corr.to_csv(OUT / "stratified_correlations.csv")
print(f"n={len(coh)}; wrote 3 CSVs to {OUT}")

for dark in (False, True):
    tag = "dark" if dark else "light"
    bg = "#1a1a19" if dark else "#fcfcfb"
    fp.stratified_prevalence_fig(prev, dark).savefig(
        FIGDIR / f"fig6_strat_prev_{tag}.png", facecolor=bg, bbox_inches="tight")
    fp.stratified_corr_heatmap(corr, dark).savefig(
        FIGDIR / f"fig7_strat_corr_{tag}.png", facecolor=bg, bbox_inches="tight")
print(f"wrote figures to {FIGDIR}")
