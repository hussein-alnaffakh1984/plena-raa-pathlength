#!/usr/bin/env python3
"""Q1 UPGRADE part 2: geometry systematic scan.
Vary Glauber inputs (sigma_nn, Woods-Saxon R & a, exit-length threshold) and the
geometry DEFINITION (A^1/3, <Npart>^1/3, exit <L>); refit n each time.
Quantifies (i) Glauber-modelling uncertainty within a definition, and
(ii) the dominant definition-choice systematic.
"""
import os, glob, re, json, yaml, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
OUT = "/tmp/out"; NEW = "/tmp/real4/HEPData-ins3123773-v1-yaml"; OLD = "/tmp/real"; PT_MIN = 8.0
A_OF = {"OO": 16, "NeNe": 20, "XeXe": 129, "PbPb": 208}; SQRTS = {"OO": 5.36, "NeNe": 5.36, "XeXe": 5.44, "PbPb": 5.02}
ORDER = ["OO", "NeNe", "XeXe", "PbPb"]; META = {"pt", "raa", "system", "A"}
apoly = np.load("/tmp/a_poly.npy"); a_of = lambda pt, s: apoly[0]*np.log(np.asarray(pt)*5.02/s)**2+apoly[1]*np.log(np.asarray(pt)*5.02/s)+apoly[2]
WS0 = {"OO": (2.608, 0.513), "NeNe": (2.80, 0.55), "XeXe": (5.40, 0.59), "PbPb": (6.62, 0.546)}; CMAX = {"OO": 1, "NeNe": 1, "XeXe": 0.80, "PbPb": 1}

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
    n = n.lower(); return "u" if "stat" in n else ("f" if any(k in n for k in ("taa", "lumi", "norm", "global")) else "p")
def cov(df, xi=4.0):
    n = len(df); C = np.zeros((n, n)); D = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    for c in comps(df):
        v = df[c].values.astype(float); k = classify(c)
        C += np.diag(v**2) if k == "u" else (np.outer(v, v) if k == "f" else np.outer(v, v)*np.exp(-D/xi))
    return C
DFS = {"OO": load_new("oo_raa_(coarse_pt_binning).yaml", "OO"), "NeNe": load_new("nene_raa_(coarse_pt_binning).yaml", "NeNe"),
       "PbPb": load_new("pbpb_raa_(coarse_pt_binning).yaml", "PbPb"), "XeXe": load_xexe()}
FIT = {}
for nm in ORDER:
    df = DFS[nm]; C = cov(df); i = np.where(df["pt"].values >= PT_MIN)[0]
    FIT[nm] = dict(pt=df["pt"].values[i], raa=df["raa"].values[i], cov=C[np.ix_(i, i)])
CH = {nm: cho_factor(FIT[nm]["cov"]) for nm in FIT}
fs = {s: (SQRTS[s]/5.02)**0.31 for s in ORDER}

# ---- fast Glauber ----
Gx = 16.0; NG = 141; xs = np.linspace(-Gx, Gx, NG); dx = xs[1]-xs[0]; X, Y = np.meshgrid(xs, xs, indexing="ij")
zs = np.linspace(-Gx, Gx, 141); dz = zs[1]-zs[0]
def thick(R, a): r = np.sqrt(X[..., None]**2+Y[..., None]**2+zs[None, None, :]**2); return (1/(1+np.exp((r-R)/a))).sum(2)*dz
def glong(s, b, R, a, sig, frac, npts=700, ndir=12, nstep=180, seed=0):
    T = thick(R, a); sh = b/(2*dx); iy, ix = np.indices(T.shape)
    TA = map_coordinates(T, [ix-sh, iy], order=1, mode="constant"); TB = map_coordinates(T, [ix+sh, iy], order=1, mode="constant")
    nc = TA*TB; npart = TA*(1-np.exp(-sig*TB))+TB*(1-np.exp(-sig*TA)); Np = npart.sum()*dx*dx
    if nc.max() <= 0: return 0., Np
    thr = frac*nc.max(); rng = np.random.default_rng(seed); fl = nc.ravel()/nc.sum(); sel = rng.choice(fl.size, size=npts, p=fl)
    px, py = X.ravel()[sel], Y.ravel()[sel]; phis = np.linspace(0, 2*np.pi, ndir, endpoint=False); ls = np.arange(1, nstep+1)*dx
    cx = px[:, None, None]+ls*np.cos(phis)[None, :, None]; cy = py[:, None, None]+ls*np.sin(phis)[None, :, None]
    rho = map_coordinates(nc, [((cx+Gx)/dx).ravel(), ((cy+Gx)/dx).ravel()], order=1, mode="constant").reshape(cx.shape)
    return float(((rho > thr).cumprod(2).sum(2)*dx).mean()), Np
def geom_ratios(sig=7.0, dR=0.0, fa=1.0, frac=0.10):
    L = {}; Npart = {}
    for s in ORDER:
        R, a = WS0[s]; R *= (1+dR); a *= fa; bmax = (2*R+4*a)*np.sqrt(CMAX[s]); bs = np.linspace(0.2, bmax, 6)
        ll = []; nn = []
        for b in bs: l, n_ = glong(s, b, R, a, sig, frac); ll.append(l); nn.append(n_)
        ll = np.array(ll); nn = np.array(nn); L[s] = np.sum(bs*ll)/np.sum(bs); Npart[s] = np.sum(bs*nn)/np.sum(bs)
    GN = {s: (Npart[s]/Npart["PbPb"])**(1/3) for s in ORDER}; GL = {s: L[s]/L["PbPb"] for s in ORDER}
    return GN, GL

def nll(th, geom):
    k, n, b = th
    if not (0 < k < 12 and 0 <= n <= 4.5 and -0.5 <= b <= 1.5): return 1e12
    v = 0.5*((n-2)/1.5)**2
    for nm in ORDER:
        dpt = k*fs[nm]*geom[nm]**n*np.power(FIT[nm]["pt"], b); r = FIT[nm]["raa"]-(FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**a_of(FIT[nm]["pt"], SQRTS[nm])
        v += 0.5*r@cho_solve(CH[nm], r)
    return v
def nfit(geom):
    r = minimize(nll, [2, 1.8, 0.3], args=(geom,), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000}); return r.x[1]

print("Geometry systematic scan (sqrt(s)-corrected fit). n = effective exponent.\n")
variations = [("nominal", dict()), ("sigma_nn=6.4", dict(sig=6.4)), ("sigma_nn=7.6", dict(sig=7.6)),
              ("R +2%", dict(dR=0.02)), ("R -2%", dict(dR=-0.02)), ("a_WS +10%", dict(fa=1.10)), ("a_WS -10%", dict(fa=0.90)),
              ("L thr=0.05", dict(frac=0.05)), ("L thr=0.15", dict(frac=0.15))]
nN = {}; nL = {}
for lab, kw in variations:
    GN, GL = geom_ratios(**kw); nN[lab] = nfit(GN); nL[lab] = nfit(GL)
    print(f"  {lab:14s}  n(Npart^1/3)={nN[lab]:.2f}   n(exit-L)={nL[lab]:.2f}")
spanN = (min(nN.values()), max(nN.values())); spanL = (min(nL.values()), max(nL.values()))
# definition systematic (A^1/3 vs Npart vs L) from results.json + this
rj = json.load(open(os.path.join(OUT, "results.json")))["bayesian"]
nA = rj["A^1/3"]["effective"]["n"][1]
print(f"\nWithin-definition Glauber uncertainty:")
print(f"  Npart^1/3: n in [{spanN[0]:.2f}, {spanN[1]:.2f}]  -> +/-{(spanN[1]-spanN[0])/2:.2f}")
print(f"  exit-<L> : n in [{spanL[0]:.2f}, {spanL[1]:.2f}]  -> +/-{(spanL[1]-spanL[0])/2:.2f}")
print(f"\nDefinition-choice systematic (dominant): A^1/3={nA:.2f}, Npart^1/3={nN['nominal']:.2f}, exit-L={nL['nominal']:.2f}")
json.dump({"npart_scan": nN, "exitL_scan": nL, "within_def_npart_halfspan": (spanN[1]-spanN[0])/2,
           "within_def_exitL_halfspan": (spanL[1]-spanL[0])/2, "def_A13": nA}, open(os.path.join(OUT, "geometry_scan.json"), "w"), indent=1)
print("\nsaved geometry_scan.json")
