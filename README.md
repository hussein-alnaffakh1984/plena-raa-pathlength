# Path-length dependence of parton energy loss across collision systems

A Bayesian and simulation-based-inference (SBI) extraction of the effective
path-length exponent of parton energy loss, ΔE ∝ ρ·Lⁿ, from CMS charged-particle
nuclear modification factors R_AA across four collision systems
(O+O, Ne+Ne, Xe+Xe, Pb+Pb), spanning mass number A = 16–208.

**Headline result:** n_eff = 1.85 ± 0.15 (stat) ± 0.05 (syst) for the
⟨N_part⟩^{1/3} geometry — radiative-dominated (n=2), consistent with a universal
exponent from O+O to Pb+Pb. Nested-sampling model selection decisively favours
radiative scaling over collisional (n=1, 2ΔlnZ ≈ −29) and strong-coupling
(n=3, 2ΔlnZ ≈ −48).

This repository contains the full, reproducible analysis pipeline: forward
model, correlated-Gaussian likelihood, dual-Glauber geometry, nested-sampling
model comparison, coverage calibration, a 160-variant sensitivity programme,
a normalizing-flow neural posterior estimator (validated by simulation-based
calibration), a calibrated Gaussian-process prediction engine, leave-one-system-
out stability tests, and a continuous evidence landscape.

## Repository layout

```
raa-pathlength/
├── src/         analysis scripts (one module per result; see mapping below)
├── results/     JSON outputs produced by the scripts
├── figures/     all paper figures (PNG/SVG)
├── paper/       manuscript (LaTeX source + compiled PDF + referee response)
└── data/        pointers to the public HEPData records (see data/README.md)
```

## Script → result map

| Script | Produces | Paper section |
|---|---|---|
| `mc_glauber.py` | `mc_glauber.json` (dual-Glauber geometry) | Methodology |
| `mc_hires.py` | `mc_glauber_hires.json` (5000-event convergence) | Sensitivity |
| `sensitivity.py`, `sens_fig.py` | `sensitivity.json`, heatmap | Sensitivity |
| `universality.py` | `universality.json` | Universality |
| `qc_sqrts.py`, `qc_geometry.py` | QC checks | Systematics |
| `jetscape.py` | `jetscape_comparison.json` | JETSCAPE comparison |
| `loso_predict.py` | `loso_predictions.json` | LOSO + predictions |
| `gp_predict.py` | `gp_predictions.json` | ML prediction engine |
| `ml_upgrades.py` | `ml_npe_sbc.json` (NPE, 1.5×10⁴) | ML cross-validation |
| `ml_npe_50k.py` | `ml_npe_sbc_50k.json` (NPE convergence) | Sensitivity |
| `evidence_qw_qgp.py` | `evidence_landscape.json`, `qw_variant.json`, `qgp_oo.json` | Model selection, Discussion |
| `reviewer_p2.py` | `reviewer_p2.json` (fluctuations, √s, widened bands) | Systematics |
| `nshift.py` | n with high-statistics geometry | Sensitivity |
| `alice_overlay.py` | `alice_overlay.json` (external check) | Outlook |
| `make_flowchart.py` | pipeline flowchart | Fig. 1 |
| `raa_kaggle_pipeline.py` | unified notebook-style pipeline | — |

## Requirements

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. Key packages: numpy, scipy, emcee, dynesty, scikit-learn,
torch, zuko, ngboost, pyyaml, matplotlib. See `requirements.txt`.

## Data

The analysis uses public CMS and ALICE records from HEPData. They are **not**
redistributed here; download instructions and record IDs are in
[`data/README.md`](data/README.md). After downloading, update the data paths at
the top of the scripts.

## How to reproduce

1. Install requirements and download the HEPData records (see `data/README.md`).
2. Run the geometry first: `python src/mc_glauber.py`.
3. Run the main pipeline / individual modules; each writes a JSON to `results/`
   and figures to `figures/`.
4. The manuscript in `paper/` cites these outputs; recompile the LaTeX with
   any standard TeX distribution (the `article`-class version compiles
   stand-alone with `pdflatex`).

## Citation

If you use this code or its results, please cite the paper (see
`CITATION.cff`).

## License

MIT (see `LICENSE`). The HEPData inputs retain their original licenses.
