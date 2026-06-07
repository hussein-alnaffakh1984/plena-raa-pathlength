#!/usr/bin/env python3
"""
(B) UNIVERSALITY TEST via Bayes factor.
Model U (universal): one exponent n for all systems (OO,NeNe,XeXe,PbPb).
Model B (broken):    n_small for light systems (OO,NeNe), n_large for heavy (XeXe,PbPb).
A decisive preference for B would signal a change of energy-loss regime between
small and large systems (e.g. QGP onset); no preference => universal scaling.
Uses sqrt(s)-corrected fit + MC-Glauber <Npart>^1/3 geometry.
"""
import os, glob, re, json, yaml, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.linalg import cho_factor, cho_solve
from dynesty import NestedSampler
OUT = "/tmp/out"; NEW = "/tmp/real4/HEPData-ins3123773-v1-yaml"; OLD = "/tmp/real"; PT_MIN = 8.0
A_OF = {"OO": 16, "NeNe": 20, "XeXe": 129, "PbPb": 208}; SQRTS = {"OO": 5.36, "NeNe": 5.36, "XeXe": 5.44, "PbPb": 5.02}
ORDER = ["OO", "NeNe", "XeXe", "PbPb"]; SMALL = {"OO", "NeNe"}; META = {"pt", "raa", "system", "A"}
ap = np.load("/tmp/a_poly.npy"); a_of = lambda pt, s: ap[0]*np.log(np.asarray(pt)*5.02/s)**2+ap[1]*np.log(np.asarray(pt)*5.02/s)+ap[2]
def parse(d):
    iv = d["independent_variables"][0]["values"]; dv = d["dependent_variables"][0]["values"]; rows = []
    for i, val in enumerate(dv):
        v = iv[i]; pt = 0.5*(float(v["low"])+float(v["high"])) if "low" in v else float(v["value"]); row = {"pt": pt, "raa": float(val["value"])}
        for e in val.get("errors", []):
            lab = (e.get("label") or "e").strip().replace(".", "").replace(",", "_").replace(" ", "")
            row[lab] = abs(float(e["symerror"])) if "symerror" in e else 0.5*(abs(float(e["asymerror"]["plus"]))+abs(float(e["asymerror"]["minus"])))
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)
def ln(fn, nm): df = parse(yaml.safe_load(open(f"{NEW}/{fn}"))); df["system"] = nm; df["A"] = A_OF[nm]; return df
def lx():
    sub = [s for s in glob.glob(OLD+"/**/submission.yaml", recursive=True) if "1692558" in s][0]; D = os.path.dirname(sub)
    F = {os.path.basename(p): open(p).read() for p in glob.glob(D+"/*.yaml")}; best = None; span = -1
    for d in yaml.safe_load_all(F["submission.yaml"]):
        if isinstance(d, dict) and "data_file" in d:
            kw = {k["name"]: k.get("values", []) for k in d.get("keywords", [])}
            if any("RAA" in str(x).upper() for x in kw.get("observables", [])):
                dd = yaml.safe_load(F[d["data_file"]]); cen = ""
                for q in dd["dependent_variables"][0].get("qualifiers", []):
                    if "CENTRALITY" in str(q.get("name", "")).upper(): cen = str(q.get("value", ""))
                mm = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", cen); sp = float(mm[0][1])-float(mm[0][0]) if mm else 100
                if sp > span: span = sp; best = dd
    return parse(best)
DFS = {"OO": ln("oo_raa_(coarse_pt_binning).yaml", "OO"), "NeNe": ln("nene_raa_(coarse_pt_binning).yaml", "NeNe"),
       "PbPb": ln("pbpb_raa_(coarse_pt_binning).yaml", "PbPb"), "XeXe": lx()}
def cl(n):
    n = n.lower(); return "u" if "stat" in n else ("f" if any(k in n for k in ("taa", "lumi", "norm", "global")) else "p")
def cov(df, xi=4.0):
    cols = [c for c in df.columns if c not in META]; n = len(df); C = np.zeros((n, n)); D = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    for c in cols:
        v = df[c].values.astype(float); k = cl(c); C += np.diag(v**2) if k == "u" else (np.outer(v, v) if k == "f" else np.outer(v, v)*np.exp(-D/xi))
    return C
FIT = {}
for nm in ORDER:
    df = DFS[nm]; C = cov(df); i = np.where(df["pt"].values >= PT_MIN)[0]; FIT[nm] = dict(pt=df["pt"].values[i], raa=df["raa"].values[i], cov=C[np.ix_(i, i)])
CH = {nm: cho_factor(FIT[nm]["cov"]) for nm in FIT}; fs = {s: (SQRTS[s]/5.02)**0.31 for s in ORDER}
GN = json.load(open(os.path.join(OUT, "mc_glauber.json")))["mc_ratios"]["Npart13"]   # MC-Glauber geometry

def chi2(k, nmap, b):
    s = 0.0
    for nm in ORDER:
        dpt = k*fs[nm]*GN[nm]**nmap[nm]*np.power(FIT[nm]["pt"], b); r = FIT[nm]["raa"]-(FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**a_of(FIT[nm]["pt"], SQRTS[nm])
        s += r@cho_solve(CH[nm], r)
    return s
def llU(th): k, n, b = th; return -0.5*chi2(k, {nm: n for nm in ORDER}, b)
def ptU(u): return np.array([12*u[0], 4.5*u[1], -0.5+2*u[2]])
def llB(th):
    k, ns, nl, b = th; nmap = {nm: (ns if nm in SMALL else nl) for nm in ORDER}; return -0.5*chi2(k, nmap, b)
def ptB(u): return np.array([12*u[0], 4.5*u[1], 4.5*u[2], -0.5+2*u[3]])
def eviz(ll, pt, nd):
    s = NestedSampler(ll, pt, nd, nlive=400, rstate=np.random.default_rng(0)); s.run_nested(print_progress=False, dlogz=0.1)
    return s.results
print("UNIVERSALITY TEST (MC-Glauber Npart^1/3, sqrt(s)-corrected)\n")
rU = eviz(llU, ptU, 3); rB = eviz(llB, ptB, 4)
ZU, ZB = rU.logz[-1], rB.logz[-1]
# posterior of n_small, n_large in broken model
import numpy as np
w = np.exp(rB.logwt-rB.logz[-1]); samp = rB.samples
ns_m = np.percentile(samp[:, 1], [16, 50, 84]); nl_m = np.percentile(samp[:, 2], [16, 50, 84])
print(f"  Model U (universal n):   logZ = {ZU:.2f}")
print(f"  Model B (broken n):      logZ = {ZB:.2f}")
print(f"  2 ln(B_BU) = {2*(ZB-ZU):+.2f}   (>0 favors broken; <0 favors universal)")
print(f"  broken-model:  n_small(OO,NeNe) = {ns_m[1]:.2f} (+{ns_m[2]-ns_m[1]:.2f}/-{ns_m[1]-ns_m[0]:.2f})"
      f"   n_large(XeXe,PbPb) = {nl_m[1]:.2f} (+{nl_m[2]-nl_m[1]:.2f}/-{nl_m[1]-nl_m[0]:.2f})")
verdict = ("DECISIVE for broken" if 2*(ZB-ZU) > 10 else "moderate for broken" if 2*(ZB-ZU) > 2
           else "no preference -> UNIVERSAL scaling OO->PbPb" if 2*(ZB-ZU) > -2 else "favors universal")
print(f"\n  Jeffreys verdict: {verdict}")
json.dump({"logZ_universal": ZU, "logZ_broken": ZB, "2lnB_BU": 2*(ZB-ZU),
           "n_small": ns_m.tolist(), "n_large": nl_m.tolist(), "verdict": verdict},
          open(os.path.join(OUT, "universality.json"), "w"), indent=1)
print("\nsaved universality.json")
