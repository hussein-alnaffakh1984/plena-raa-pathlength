PLENA — Path-Length R_AA Analysis Code
Reproducible analysis code and an interactive calculator (PLENA) for studying how
parton energy loss, ΔE ∝ ρ·Lⁿ, scales with collision-system size, using public
CMS charged-particle nuclear-modification factors (R_AA). The repository provides
the full computational pipeline and a browser-based tool; the scientific write-up
is described in a separate manuscript (in preparation) and is not included here.
What this repository contains
`src/` — analysis scripts (geometry, likelihood, model comparison, sensitivity,
machine-learning cross-validation); one module per computational result.
`results/` — JSON outputs produced by the scripts.
`figures/` — generated plots (PNG).
`data/` — pointers to the public HEPData records used as input (see `data/README.md`).
`index.html` — PLENA, a standalone interactive R_AA calculator (open in any browser).
The interactive calculator (PLENA)
`index.html` is a self-contained tool: choose a collision system, type or drag the
model parameters (n, κ, β), and compare the computed R_AA(pT) against the public
CMS data. It reports a live goodness-of-fit, supports auto-fitting, accepts
user-supplied data (CSV), and exports the computed curve. Open it locally or host
it with any static-site service.
Script → output map
Script	Output
`mc_glauber.py`	`mc_glauber.json` (Glauber geometry)
`mc_hires.py`	`mc_glauber_hires.json` (high-statistics geometry)
`sensitivity.py`, `sens_fig.py`	`sensitivity.json`, heatmap
`universality.py`	`universality.json`
`qc_sqrts.py`, `qc_geometry.py`	quality-control checks
`jetscape.py`	`jetscape_comparison.json`
`loso_predict.py`	`loso_predictions.json`
`gp_predict.py`	`gp_predictions.json`
`ml_upgrades.py`, `ml_npe_50k.py`	`ml_npe_sbc.json`, `ml_npe_sbc_50k.json`
`evidence_qw_qgp.py`	`evidence_landscape.json`, `qw_variant.json`, `qgp_oo.json`
`alice_overlay.py`	`alice_overlay.json`
`make_flowchart.py`	pipeline flowchart
Requirements
```bash
pip install -r requirements.txt
```
Python ≥ 3.10. Key packages: numpy, scipy, emcee, dynesty, scikit-learn, torch,
zuko, ngboost, pyyaml, matplotlib.
Data
Input R_AA datasets are public CMS and ALICE records on HEPData. They are not
redistributed here; record IDs and download instructions are in
`data/README.md`. Update the data paths at the top of the scripts
after downloading.
How to reproduce
Install requirements and download the HEPData records (see `data/README.md`).
Run the geometry first: `python src/mc_glauber.py`.
Run the individual modules; each writes a JSON to `results/` and figures to `figures/`.
Citation
A citation entry is provided in `CITATION.cff`. When the associated paper is
published, please cite it (a reference will be added here).
License
MIT (see `LICENSE`). The HEPData inputs retain their original licenses.
