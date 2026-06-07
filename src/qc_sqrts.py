#!/usr/bin/env python3
"""Q1 UPGRADE part 1: sqrt(s) correction + data-driven baseline.
 - local spectral index a(pT) from pp 5.02 TeV (replaces single fixed a0)
 - xT-scaling of a for each system's sqrt(s):  a_sys(pT) = a_502(pT * 5.02/sqrt(s))
 - medium-density sqrt(s) factor f_s = (sqrt(s)/5.02)^0.31  (dN/deta growth)
 Refit cross-system n: (baseline a0) vs (local a(pT)) vs (sqrt(s)-corrected). Compare.
"""
import os, glob, re, json, yaml, warnings, numpy as np, pandas as pd, emcee
warnings.filterwarnings("ignore")
from scipy.linalg import cho_factor, cho_solve
OUT = "/tmp/out"; NEW = "/tmp/real4/HEPData-ins3123773-v1-yaml"; OLD = "/tmp/real"; PT_MIN = 8.0
A_OF = {"OO": 16, "NeNe": 20, "XeXe": 129, "PbPb": 208}
SQRTS = {"OO": 5.36, "NeNe": 5.36, "XeXe": 5.44, "PbPb": 5.02}     # TeV per system
META = {"pt", "raa", "system", "A"}
apoly = np.load("/tmp/a_poly.npy")
def a_of(pt, sqrts):  # xT-scaled local spectral index
    l = np.log(np.asarray(pt)*5.02/sqrts); return apoly[0]*l*l + apoly[1]*l + apoly[2]

def parse(d):
    iv = d["independent_variables"][0]["values"]; dv = d["dependent_variables"][0]["values"]; rows = []
    for i, val in enumerate(dv):
        v = iv[i]; pt = 0.5*(float(v["low"])+float(v["high"])) if "low" in v else float(v["value"]); row = {"pt": pt, "raa": float(val["value"])}
        for e in val.get("errors", []):
            lab = (e.get("label") or "e").strip().replace(".", "").replace(",", "_").replace(" ", "")
            row[lab] = abs(float(e["symerror"])) if "symerror" in e else 0.5*(abs(float(e["asymerror"]["plus"]))+abs(float(e["asymerror"]["minus"])))
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)
def load_new(fn, nm): df = parse(yaml.safe_load(open(f"{NEW}/{fn}"))); df["system"] = nm; df["A"] = A_OF[nm]; return df
def load_xexe():
    sub = [s for s in glob.glob(OLD+"/**/submission.yaml", recursive=True) if "1692558" in s][0]; D = os.path.dirname(sub)
    F = {os.path.basename(p): open(p).read() for p in glob.glob(D+"/*.yaml")}; best = None; span = -1
    for d in yaml.safe_load_all(F["submission.yaml"]):
        if isinstance(d, dict) and "data_file" in d:
            kw = {k["name"]: k.get("values", []) for k in d.get("keywords", [])}
            if any("RAA" in str(x).upper() for x in kw.get("observables", [])):
                dd = yaml.safe_load(F[d["data_file"]]); cen = "n/a"
                for dv in dd.get("dependent_variables", []):
                    for q in dv.get("qualifiers", []):
                        if "CENTRALITY" in str(q.get("name", "")).upper(): cen = str(q.get("value", ""))
                m = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", cen.lower()); sp = float(m[0][1])-float(m[0][0]) if m else 100
                if sp > span: span = sp; best = dd
    df = parse(best); df["system"] = "XeXe"; df["A"] = 129; return df
def comps(df): return [c for c in df.columns if c not in META]
def classify(n):
    n = n.lower()
    if "stat" in n: return "u"
    if any(k in n for k in ("taa", "lumi", "norm", "global")): return "f"
    return "p"
def cov(df, xi=4.0):
    n = len(df); C = np.zeros((n, n)); D = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    for c in comps(df):
        v = df[c].values.astype(float); k = classify(c)
        C += np.diag(v**2) if k == "u" else (np.outer(v, v) if k == "f" else np.outer(v, v)*np.exp(-D/xi))
    return C
ORDER = ["OO", "NeNe", "XeXe", "PbPb"]
DFS = {"OO": load_new("oo_raa_(coarse_pt_binning).yaml", "OO"), "NeNe": load_new("nene_raa_(coarse_pt_binning).yaml", "NeNe"),
       "PbPb": load_new("pbpb_raa_(coarse_pt_binning).yaml", "PbPb"), "XeXe": load_xexe()}
FIT = {}
for nm in ORDER:
    df = DFS[nm]; C = cov(df); i = np.where(df["pt"].values >= PT_MIN)[0]
    FIT[nm] = dict(pt=df["pt"].values[i], raa=df["raa"].values[i], cov=C[np.ix_(i, i)])
CH = {nm: cho_factor(FIT[nm]["cov"]) for nm in FIT}
geo = json.load(open(os.path.join(OUT, "results.json")))["geometry"]
GN = {s: (geo["Npart"][s]/geo["Npart"]["PbPb"])**(1/3) for s in ORDER}
a0_fixed = 6.337

def fit(mode, geom=GN, nsteps=7000, nw=48, seed=0):
    """mode: 'baseline'(single a0 prior), 'local_a'(a(pT) at 5.02 for all), 'sqrts'(xT-scaled a + f_s medium)."""
    rng = np.random.default_rng(seed)
    fs = {s: (SQRTS[s]/5.02)**0.31 for s in ORDER} if mode == "sqrts" else {s: 1.0 for s in ORDER}
    def amodel(nm, pt):
        if mode == "baseline": return None        # a is a fit param
        if mode == "local_a": return a_of(pt, 5.02)
        return a_of(pt, SQRTS[nm])                  # sqrts: xT-scaled
    def loglike(th):
        if mode == "baseline": k, n, b, a = th
        else: k, n, b = th
        if not (0 < k < 12 and 0 <= n <= 4.5 and -0.5 <= b <= 1.5): return -np.inf
        if mode == "baseline" and not (3 < a < 9): return -np.inf
        pr = -0.5*((n-2)/1.5)**2 + (-0.5*((a-a0_fixed)/0.20)**2 if mode == "baseline" else 0.0)
        ll = 0
        for nm in ORDER:
            av = a if mode == "baseline" else amodel(nm, FIT[nm]["pt"])
            dpt = k*fs[nm]*geom[nm]**n*np.power(FIT[nm]["pt"], b)
            r = FIT[nm]["raa"] - (FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**av
            ll += -0.5*r@cho_solve(CH[nm], r)
        return pr+ll
    nd = 4 if mode == "baseline" else 3
    p0 = (np.array([2., 1.8, .3, a0_fixed])[:nd]) + 0.02*rng.standard_normal((nw, nd))
    s = emcee.EnsembleSampler(nw, nd, loglike); s.run_mcmc(p0, nsteps, progress=False)
    f = s.get_chain(discard=int(nsteps*0.4), thin=10, flat=True); nq = np.percentile(f[:, 1], [16, 50, 84]); thm = np.median(f, 0)
    chi2 = 0; npts = 0
    for nm in ORDER:
        av = thm[3] if mode == "baseline" else amodel(nm, FIT[nm]["pt"])
        dpt = thm[0]*fs[nm]*geom[nm]**thm[1]*np.power(FIT[nm]["pt"], thm[2])
        r = FIT[nm]["raa"] - (FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**av; chi2 += r@cho_solve(CH[nm], r); npts += len(r)
    return nq, chi2/(npts-nd)

print("sqrt(s) per system:", SQRTS)
print("\nCross-system effective n (Npart^1/3) under baseline / local-a(pT) / sqrt(s)-corrected:\n")
res = {}
for mode, lab in [("baseline", "baseline (single a0=6.34)"), ("local_a", "local a(pT) [data-driven baseline]"), ("sqrts", "sqrt(s)-corrected (xT-scaled a + f_s)")]:
    nq, cd = fit(mode); res[mode] = nq.tolist()
    print(f"  {lab:42s}: n={nq[1]:.2f} (+{nq[2]-nq[1]:.2f}/-{nq[1]-nq[0]:.2f})  chi2/dof={cd:.2f}")
dn = abs(res["sqrts"][1]-res["local_a"][1])
print(f"\n=> sqrt(s) correction shifts n by {dn:.3f}  (systematic from energy mismatch)")
print(f"=> baseline a0 vs local a(pT) shifts n by {abs(res['local_a'][1]-res['baseline'][1]):.3f}")
json.dump({"sqrts_per_system": SQRTS, "n_baseline": res["baseline"], "n_local_a": res["local_a"],
           "n_sqrts_corrected": res["sqrts"], "dn_sqrts": dn}, open(os.path.join(OUT, "sqrts_correction.json"), "w"), indent=1)
print("\nsaved sqrts_correction.json")
