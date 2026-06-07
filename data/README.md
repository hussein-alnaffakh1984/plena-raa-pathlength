# Data sources (public HEPData records)

This analysis uses publicly available records from HEPData. Download each as
YAML and place under the paths expected at the top of the scripts in `src/`.

| System / use | Experiment | HEPData record |
|---|---|---|
| O+O, Ne+Ne, Pb+Pb R_AA (5.36 / 5.02 TeV) | CMS (HIN-25-014) | ins3123773 |
| Xe+Xe R*_AA (5.44 TeV) | CMS | ins1692558 |
| Pb+Pb pp reference + centrality R_AA (5.02 TeV) | CMS | ins1496050 |
| ALICE Xe+Xe R_AA (5.44 TeV) — external check only | ALICE (PLB 788) | ins1672790 |

Download, e.g.:
  https://www.hepdata.net/record/ins3123773  (choose "Download All" → YAML)

The CMS records provide the fitted systems; the ALICE record is used only as an
independent cross-experiment validation (NOT in the fit). The data-driven
spectral index a(p_T) is derived from the CMS pp spectrum (ins1496050).

Note: the scripts reference local paths such as /tmp/real4/... and /tmp/real/...
Update these to your local download locations.
