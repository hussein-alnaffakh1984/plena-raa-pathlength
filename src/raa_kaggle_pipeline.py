# %% [markdown]
# # Path-length dependence of parton energy loss across collision systems
# ## A Bayesian + probabilistic-ML analysis of CMS charged-particle $R_{AA}$
#
# **End-to-end, reproducible pipeline (Kaggle-ready).** Extracts the system-size
# (path-length) exponent $n$ in $\Delta E \propto \rho\,L^{n}$ from CMS $R_{AA}$ of
# **O+O, Ne+Ne, Xe+Xe, Pb+Pb**, with:
# 1. correlated covariance + fixed data-driven spectral index $a(p_T)$
# 2. optical **and** Monte-Carlo Glauber geometry (independent cross-check)
# 3. Bayesian extraction (effective + density-normalized "pure" exponent)
# 4. Bayes-factor model selection ($n=1$ collisional / $2$ radiative / $3$ strong-coupling)
# 5. coverage/closure calibration test
# 6. **universality test** (single vs broken exponent, O+O$\to$Pb+Pb)
# 7. $\sqrt{s}$ correction + full systematic budget
# 8. **Machine learning**: probabilistic emulator + leave-one-system-out,
#    and **neural posterior estimation (normalizing flow) with simulation-based calibration**
#
# Heavy stages are gated by the flags in the CONFIG cell.
#
# **Data**: HEPData ins3123773 (CMS-HIN-25-014; OO/NeNe/PbPb/pPb) + ins1692558 (XeXe)
# + ins1496050 (PbPb pp reference & centrality). Upload these as a Kaggle dataset and
# set `DATA_ROOT`.

# %% [markdown]
# ## Cell 0 — Setup, dependencies, CONFIG flags

# %%
import os, sys, glob, re, json, time, warnings, subprocess
warnings.filterwarnings("ignore")

def _pip(pkgs):
    for p in pkgs:
        mod = p.split("==")[0].replace("-", "_")
        try:
            __import__(mod if mod != "scikit_learn" else "sklearn")
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)
# Kaggle usually has numpy/scipy/pandas/sklearn/matplotlib. These may need install:
_pip(["emcee", "dynesty", "ngboost"])
# torch + zuko enable the normalizing-flow NPE (Cell 8B). If unavailable, that cell is skipped gracefully.
try:
    import torch, zuko  # noqa
    HAVE_FLOW = True
except Exception:
    try:
        _pip(["torch", "zuko"]); import torch, zuko  # noqa
        HAVE_FLOW = True
    except Exception:
        HAVE_FLOW = False

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates
from scipy.stats import norm
import emcee

# ----------------------------- CONFIG ---------------------------------------
DATA_ROOT = os.environ.get("RAA_DATA", "/kaggle/input")   # folder holding the HEPData archives
OUT       = os.environ.get("RAA_OUT", "/kaggle/working/out"); os.makedirs(OUT, exist_ok=True)
PT_MIN    = 8.0
SIGNN     = 7.0                      # nucleon-nucleon inelastic cross section [fm^2]
# ---- run flags: turn heavy stages on/off (env-overridable; default ALL ON for Kaggle) ----
_flag = lambda k, d=True: os.environ.get(k, "1" if d else "0") == "1"
RUN_OPTICAL_GLAUBER = _flag("RUN_OPTICAL_GLAUBER")
RUN_MC_GLAUBER      = _flag("RUN_MC_GLAUBER")        # ~3-6 min
RUN_BAYES_FACTORS   = _flag("RUN_BAYES_FACTORS")     # nested sampling, ~3-5 min
RUN_COVERAGE        = _flag("RUN_COVERAGE")          # ~2-3 min
RUN_UNIVERSALITY    = _flag("RUN_UNIVERSALITY")      # ~2-3 min
RUN_SQRTS           = _flag("RUN_SQRTS")
RUN_GEOM_SCAN       = _flag("RUN_GEOM_SCAN")         # ~4-6 min
RUN_ML_EMULATOR     = _flag("RUN_ML_EMULATOR")       # GP/NGBoost/ensemble, fast
RUN_ML_NPE_FLOW     = _flag("RUN_ML_NPE_FLOW") and HAVE_FLOW   # normalizing-flow NPE + SBC, ~3-5 min
SEED = 0; np.random.seed(SEED)

A_OF   = {"OO": 16, "NeNe": 20, "XeXe": 129, "PbPb": 208}
SQRTS  = {"OO": 5.36, "NeNe": 5.36, "XeXe": 5.44, "PbPb": 5.02}   # TeV
ORDER  = ["OO", "NeNe", "XeXe", "PbPb"]
COL    = {"PbPb": "#1f4e79", "XeXe": "#c0504d", "OO": "#2e7d32", "NeNe": "#8e44ad"}
RESULTS = {}
def savefig(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("  saved fig:", name)
print("flow(NPE) available:", HAVE_FLOW)

# %% [markdown]
# ## Cell 1 — Locate HEPData records & generic table parser
# Finds the three records anywhere under `DATA_ROOT` (robust to Kaggle's nesting).

# %%
def find_record(token):
    hits = [s for s in glob.glob(DATA_ROOT + "/**/submission.yaml", recursive=True) if token in s]
    if not hits:
        raise FileNotFoundError(f"record {token} not found under {DATA_ROOT}")
    return os.path.dirname(hits[0])
import yaml
def parse_table(d):
    iv = d["independent_variables"][0]["values"]; dv = d["dependent_variables"][0]["values"]; rows = []
    for i, val in enumerate(dv):
        v = iv[i]; pt = 0.5*(float(v["low"])+float(v["high"])) if "low" in v else float(v["value"])
        row = {"pt": pt, "raa": float(val["value"])}
        for e in val.get("errors", []):
            lab = (e.get("label") or "e").strip().replace(".", "").replace(",", "_").replace(" ", "")
            row[lab] = abs(float(e["symerror"])) if "symerror" in e else \
                       0.5*(abs(float(e["asymerror"]["plus"]))+abs(float(e["asymerror"]["minus"])))
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)
NEW = find_record("ins3123773"); XE = find_record("ins1692558"); PB = find_record("ins1496050")
print("records:", os.path.basename(NEW), os.path.basename(XE), os.path.basename(PB))

# %% [markdown]
# ## Cell 2 — Load 4-system $R_{AA}$, build correlation-length covariance
# Uncertainty model: stat = uncorrelated; norm/lumi/TAA = fully correlated;
# remaining syst = correlated with length $\xi=4$ bins. Restrict to $p_T\ge 8$ GeV.

# %%
META = {"pt", "raa", "system", "A"}
def load_new(fn, nm):
    df = parse_table(yaml.safe_load(open(f"{NEW}/{fn}"))); df["system"] = nm; df["A"] = A_OF[nm]; return df
def load_centrality_or_inclusive(record_dir, token_inclusive=True):
    F = {os.path.basename(p): open(p).read() for p in glob.glob(record_dir + "/*.yaml")}
    best = None; span = -1
    for d in yaml.safe_load_all(F["submission.yaml"]):
        if isinstance(d, dict) and "data_file" in d:
            kw = {k["name"]: k.get("values", []) for k in d.get("keywords", [])}
            if any("RAA" in str(x).upper() for x in kw.get("observables", [])):
                dd = yaml.safe_load(F[d["data_file"]]); cen = "n/a"
                for q in dd["dependent_variables"][0].get("qualifiers", []):
                    if "CENTRALITY" in str(q.get("name", "")).upper(): cen = str(q.get("value", ""))
                m = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", cen.lower()); sp = float(m[0][1])-float(m[0][0]) if m else 100
                if sp > span: span = sp; best = dd
    return parse_table(best)
def comps(df): return [c for c in df.columns if c not in META]
def classify(n):
    n = n.lower()
    if "stat" in n: return "u"
    if any(k in n for k in ("taa", "lumi", "norm", "global")): return "f"
    return "p"
def build_cov(df, xi=4.0):
    n = len(df); C = np.zeros((n, n)); D = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    for c in comps(df):
        v = df[c].values.astype(float); k = classify(c)
        C += np.diag(v**2) if k == "u" else (np.outer(v, v) if k == "f" else np.outer(v, v)*np.exp(-D/xi))
    return C
def restrict(df):
    C = build_cov(df); i = np.where(df["pt"].values >= PT_MIN)[0]
    return dict(pt=df["pt"].values[i], A=int(df["A"].iloc[0]), raa=df["raa"].values[i], cov=C[np.ix_(i, i)])
DFS = {"OO": load_new("oo_raa_(coarse_pt_binning).yaml", "OO"),
       "NeNe": load_new("nene_raa_(coarse_pt_binning).yaml", "NeNe"),
       "PbPb": load_new("pbpb_raa_(coarse_pt_binning).yaml", "PbPb"),
       "XeXe": load_centrality_or_inclusive(XE).assign(system="XeXe", A=A_OF["XeXe"])}
FIT = {nm: restrict(DFS[nm]) for nm in ORDER}; CH = {nm: cho_factor(FIT[nm]["cov"]) for nm in FIT}
for nm in ORDER: print(f"  {nm:5s} A={A_OF[nm]:3d}  n(>=8)={len(FIT[nm]['pt']):2d}")

# %% [markdown]
# ## Cell 3 — Data-driven baseline: local spectral index $a(p_T)$ + $\sqrt{s}$ via $x_T$-scaling
# $a(p_T)=-d\ln(dN/dp_T)/d\ln p_T$ from the measured pp spectrum (5.02 TeV);
# per-system $a$ uses $x_T$-scaling $a_{\rm sys}(p_T)=a_{502}(p_T\cdot 5.02/\sqrt{s})$.

# %%
F_PB = {os.path.basename(p): open(p).read() for p in glob.glob(PB + "/*.yaml")}
pp = None
for d in yaml.safe_load_all(F_PB["submission.yaml"]):
    if isinstance(d, dict) and "data_file" in d:
        kw = {k["name"]: k.get("values", []) for k in d.get("keywords", [])}
        if "P P" in str(kw.get("reactions", "")) and "ED3N" in str(kw.get("observables", "")).upper():
            pp = yaml.safe_load(F_PB[d["data_file"]]); break
ivp = pp["independent_variables"][0]["values"]; dvp = pp["dependent_variables"][0]["values"]
ptp = np.array([0.5*(float(v["low"])+float(v["high"])) if "low" in v else float(v["value"]) for v in ivp])
spec = np.array([float(v["value"]) for v in dvp]); m = (ptp >= 4) & (spec > 0); ptp, spec = ptp[m], spec[m]
ai = -np.gradient(np.log(2*np.pi*ptp*spec), np.log(ptp))
mm = (ptp >= 6) & (ptp <= 150); APOLY = np.polyfit(np.log(ptp[mm]), ai[mm], 2)
def a_of(pt, sqrts): l = np.log(np.asarray(pt)*5.02/sqrts); return APOLY[0]*l*l+APOLY[1]*l+APOLY[2]
print("  a(pT) at 5.02 TeV: a(10)=%.2f a(30)=%.2f a(100)=%.2f" % (a_of(10, 5.02), a_of(30, 5.02), a_of(100, 5.02)))

# %% [markdown]
# ## Cell 4 — Optical Glauber geometry ($\langle L\rangle$, $\langle N_{part}\rangle$, area, density)

# %%
WS = {"OO": (2.608, 0.513), "NeNe": (2.80, 0.55), "XeXe": (5.40, 0.59), "PbPb": (6.62, 0.546)}
CMAX = {"OO": 1, "NeNe": 1, "XeXe": 0.80, "PbPb": 1}
Gx = 16.0; NG = 161; _xs = np.linspace(-Gx, Gx, NG); _dx = _xs[1]-_xs[0]; _X, _Y = np.meshgrid(_xs, _xs, indexing="ij")
def _thick(R, a):
    zs = np.linspace(-Gx, Gx, 161); dz = zs[1]-zs[0]
    r = np.sqrt(_X[..., None]**2+_Y[..., None]**2+zs[None, None, :]**2); return (1/(1+np.exp((r-R)/a))).sum(2)*dz
def _opt(s, b, frac=0.10, npts=1200, ndir=16, nstep=200):
    R, a = WS[s]; T = _thick(R, a); sh = b/(2*_dx); iy, ix = np.indices(T.shape)
    TA = map_coordinates(T, [ix-sh, iy], order=1, mode="constant"); TB = map_coordinates(T, [ix+sh, iy], order=1, mode="constant")
    nc = TA*TB; npart = TA*(1-np.exp(-SIGNN*TB))+TB*(1-np.exp(-SIGNN*TA)); Np = npart.sum()*_dx*_dx
    w = npart/(npart.sum()+1e-12); mx = (w*_X).sum(); my = (w*_Y).sum()
    S = 4*np.pi*np.sqrt((w*(_X-mx)**2).sum()*(w*(_Y-my)**2).sum())
    if nc.max() <= 0: return 0., Np, S
    thr = frac*nc.max(); rng = np.random.default_rng(0); fl = nc.ravel()/nc.sum(); sel = rng.choice(fl.size, size=npts, p=fl)
    px, py = _X.ravel()[sel], _Y.ravel()[sel]; phis = np.linspace(0, 2*np.pi, ndir, endpoint=False); ls = np.arange(1, nstep+1)*_dx
    cx = px[:, None, None]+ls*np.cos(phis)[None, :, None]; cy = py[:, None, None]+ls*np.sin(phis)[None, :, None]
    rho = map_coordinates(nc, [((cx+Gx)/_dx).ravel(), ((cy+Gx)/_dx).ravel()], order=1, mode="constant").reshape(cx.shape)
    return float(((rho > thr).cumprod(2).sum(2)*_dx).mean()), Np, S
def optical_geometry():
    L = {}; Np = {}; S = {}
    for s in WS:
        R, a = WS[s]; bmax = (2*R+4*a)*np.sqrt(CMAX[s]); bs = np.linspace(0.1, bmax, 8); ll = []; nn = []; ss = []
        for b in bs: l, n_, s_ = _opt(s, b); ll.append(l); nn.append(n_); ss.append(s_)
        ll, nn, ss = map(np.array, (ll, nn, ss)); L[s] = np.sum(bs*ll)/np.sum(bs); Np[s] = np.sum(bs*nn)/np.sum(bs); S[s] = np.sum(bs*nn*ss)/np.sum(bs*nn)
    return L, Np, S
if RUN_OPTICAL_GLAUBER:
    Lo, Npo, So = optical_geometry()
    GEO_OPT = {"Npart13": {s: (Npo[s]/Npo["PbPb"])**(1/3) for s in WS}, "L": {s: Lo[s]/Lo["PbPb"] for s in WS},
               "A13": {s: (A_OF[s]/208)**(1/3) for s in WS}}
    RHO = {s: (Npo[s]/So[s])/(Npo["PbPb"]/So["PbPb"]) for s in WS}
    RESULTS["optical_geometry"] = {"L": Lo, "Npart": Npo, "S": So, "rho": RHO, "ratios": GEO_OPT}
    print("  optical Npart^1/3:", {s: round(GEO_OPT["Npart13"][s], 3) for s in ORDER})

# %% [markdown]
# ## Cell 5 — Monte-Carlo Glauber (independent cross-check, pure Python, no ROOT)
# Samples nucleons from Woods-Saxon, applies the $\sigma_{nn}$ collision criterion,
# computes $\langle N_{part}\rangle,\langle N_{coll}\rangle$, area, eccentricity, exit $\langle L\rangle$.

# %%
def _sample_nucleus(R, a, A, dmin=0.4):
    rmax = R+8*a; pos = np.empty((A, 3)); got = 0; tries = 0
    while got < A and tries < 200000:
        r = rmax*np.cbrt(np.random.uniform(size=4*(A-got)))
        r = r[np.random.uniform(size=len(r)) < 1.0/(1.0+np.exp((r-R)/a))]
        if len(r) == 0: tries += 1; continue
        ct = np.random.uniform(-1, 1, len(r)); ph = np.random.uniform(0, 2*np.pi, len(r)); st = np.sqrt(1-ct*ct)
        for c in np.column_stack([r*st*np.cos(ph), r*st*np.sin(ph), r*ct]):
            if got == 0 or np.all(np.sum((pos[:got]-c)**2, 1) > dmin*dmin):
                pos[got] = c; got += 1
                if got >= A: break
        tries += 1
    pos[:, :2] -= pos[:, :2].mean(0); return pos
def _mc_event(s, b):
    R, a = WS[s]; A = A_OF[s]; NA = _sample_nucleus(R, a, A); NB = _sample_nucleus(R, a, A)
    NA[:, 0] -= b/2; NB[:, 0] += b/2; d2 = SIGNN/np.pi
    hit = ((NA[:, 0][:, None]-NB[:, 0][None, :])**2+(NA[:, 1][:, None]-NB[:, 1][None, :])**2) <= d2
    Ncoll = int(hit.sum()); pA = hit.any(1); pB = hit.any(0); Npart = int(pA.sum()+pB.sum())
    if Npart < 2: return None
    px = np.concatenate([NA[pA, 0], NB[pB, 0]]); py = np.concatenate([NA[pA, 1], NB[pB, 1]]); px -= px.mean(); py -= py.mean()
    sx2 = np.mean(px*px); sy2 = np.mean(py*py); S = 4*np.pi*np.sqrt(max(sx2*sy2, 1e-9))
    ax_, ay_ = np.sqrt(sx2), np.sqrt(sy2); phis = np.linspace(0, 2*np.pi, 12, endpoint=False); Ls = []
    for i in np.random.choice(len(px), size=min(60, len(px)), replace=False):
        x0, y0 = px[i], py[i]
        for ph in phis:
            cx, cy = np.cos(ph), np.sin(ph); Ax = (cx/(2*ax_))**2+(cy/(2*ay_))**2
            Bx = 2*(x0*cx/(2*ax_)**2+y0*cy/(2*ay_)**2); Cx = (x0/(2*ax_))**2+(y0/(2*ay_))**2-1; disc = Bx*Bx-4*Ax*Cx
            if disc > 0 and Ax > 0:
                t = (-Bx+np.sqrt(disc))/(2*Ax)
                if t > 0: Ls.append(t)
    return Npart, Ncoll, S, np.mean(Ls) if Ls else 0.0
def mc_geometry(nevt=500):
    out = {}
    for s in ORDER:
        R, a = WS[s]; bmax = (2*R+5*a)*np.sqrt(CMAX[s]); rows = []
        for _ in range(nevt):
            e = _mc_event(s, bmax*np.sqrt(np.random.uniform()))
            if e: rows.append(e)
        arr = np.array(rows); out[s] = dict(Npart=arr[:, 0].mean(), Ncoll=arr[:, 1].mean(), S=arr[:, 2].mean(), L=arr[:, 3].mean())
    return out
if RUN_MC_GLAUBER:
    t0 = time.time(); MC = mc_geometry(500)
    GEO_MC = {"Npart13": {s: (MC[s]["Npart"]/MC["PbPb"]["Npart"])**(1/3) for s in ORDER}, "L": {s: MC[s]["L"]/MC["PbPb"]["L"] for s in ORDER}}
    RESULTS["mc_glauber"] = {"abs": {s: {k: float(MC[s][k]) for k in MC[s]} for s in MC}, "ratios": GEO_MC}
    if RUN_OPTICAL_GLAUBER:
        dN = max(abs(GEO_MC["Npart13"][s]-GEO_OPT["Npart13"][s]) for s in ORDER)
        RESULTS["mc_glauber"]["max_diff_Npart13_vs_optical"] = float(dN)
        print(f"  MC <Npart> abs:", {s: round(MC[s]["Npart"]) for s in ORDER}, f" | max|MC-opt| Npart^1/3 = {dN:.3f}  ({time.time()-t0:.0f}s)")
GEO = GEO_MC if RUN_MC_GLAUBER else (GEO_OPT if RUN_OPTICAL_GLAUBER else None)   # default geometry for the fits
if GEO is None:
    GEO = {"Npart13": {s: (A_OF[s]/208)**(1/3) for s in ORDER}}; RHO = {s: 1.0 for s in ORDER}
    print("  (no Glauber run: using A^1/3 fallback geometry)")

# %% [markdown]
# ## Cell 6 — Bayesian extraction (effective & density-normalized "pure" exponent)
# Forward model $\Delta p_T=\kappa\,(\rho)\,G^{n}\,p_T^{\beta}$, $R_{AA}=(p_T/(p_T+\Delta p_T))^{a(p_T)}$,
# correlated-Gaussian likelihood, emcee. $\sqrt{s}$ factor $f_s=(\sqrt{s}/5.02)^{0.31}$ optional.

# %%
def raa_model(pt, rh, G, k, n, b, a, use_rho): return (pt/(pt + k*(rh if use_rho else 1.0)*G**n*np.power(pt, b)))**a
def mcmc_fit(geom, use_rho=False, sqrts_corr=True, nsteps=7000, nw=48, seed=0):
    rng = np.random.default_rng(seed); fs = {s: (SQRTS[s]/5.02)**0.31 if sqrts_corr else 1.0 for s in ORDER}
    rho = RHO if (use_rho and "RHO" in globals()) else {s: 1.0 for s in ORDER}
    def logp(th):
        k, n, b = th
        if not (0 < k < 12 and 0 <= n <= 4.5 and -0.5 <= b <= 1.5): return -np.inf
        ll = -0.5*((n-2)/1.5)**2
        for nm in ORDER:
            dpt = k*fs[nm]*(rho[nm] if use_rho else 1.0)*geom[nm]**n*np.power(FIT[nm]["pt"], b)
            r = FIT[nm]["raa"] - (FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**a_of(FIT[nm]["pt"], SQRTS[nm])
            ll += -0.5*r@cho_solve(CH[nm], r)
        return ll
    p0 = np.array([2., 1.8, .3]) + 0.02*rng.standard_normal((nw, 3))
    s = emcee.EnsembleSampler(nw, 3, logp); s.run_mcmc(p0, nsteps, progress=False)
    f = s.get_chain(discard=int(nsteps*0.4), thin=10, flat=True); nq = np.percentile(f[:, 1], [16, 50, 84]); thm = np.median(f, 0)
    chi2 = 0; npts = 0
    for nm in ORDER:
        dpt = thm[0]*fs[nm]*(rho[nm] if use_rho else 1.0)*geom[nm]**thm[1]*np.power(FIT[nm]["pt"], thm[2])
        r = FIT[nm]["raa"]-(FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**a_of(FIT[nm]["pt"], SQRTS[nm]); chi2 += r@cho_solve(CH[nm], r); npts += len(r)
    return dict(chain=f, nq=nq.tolist(), thm=thm.tolist(), chi2dof=chi2/(npts-3))
BAY = {}
for gk in GEO:
    BAY[gk] = {"effective": mcmc_fit(GEO[gk], False)}
    if "RHO" in globals(): BAY[gk]["pure"] = mcmc_fit(GEO[gk], True)
    e = BAY[gk]["effective"]; msg = f"  [{gk:8s}] eff n={e['nq'][1]:.2f} (+{e['nq'][2]-e['nq'][1]:.2f}/-{e['nq'][1]-e['nq'][0]:.2f}) chi2/dof={e['chi2dof']:.2f}"
    if "pure" in BAY[gk]: p = BAY[gk]["pure"]; msg += f" | pure n={p['nq'][1]:.2f}"
    print(msg)
np.save(os.path.join(OUT, "chain_effective.npy"), BAY[list(GEO)[0]]["effective"]["chain"])
RESULTS["bayesian"] = {gk: {k: {"n": BAY[gk][k]["nq"], "chi2dof": BAY[gk][k]["chi2dof"]} for k in BAY[gk]} for gk in BAY}

# %% [markdown]
# ## Cell 7A — Bayes-factor model selection ($n=1$ vs $2$ vs $3$)

# %%
if RUN_BAYES_FACTORS:
    from dynesty import NestedSampler
    GBF = GEO["Npart13"]; fs = {s: (SQRTS[s]/5.02)**0.31 for s in ORDER}
    def ll_fixed(th, nfix):
        k, b = th; s = 0.0
        for nm in ORDER:
            dpt = k*fs[nm]*GBF[nm]**nfix*np.power(FIT[nm]["pt"], b); r = FIT[nm]["raa"]-(FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**a_of(FIT[nm]["pt"], SQRTS[nm]); s += -0.5*r@cho_solve(CH[nm], r)
        return s
    def Z(nfix):
        smp = NestedSampler(lambda th: ll_fixed(th, nfix), lambda u: np.array([12*u[0], -0.5+2*u[1]]), 2, nlive=300, rstate=np.random.default_rng(0))
        smp.run_nested(print_progress=False, dlogz=0.3); return float(smp.results.logz[-1])
    zs = {n: Z(float(n)) for n in (1, 2, 3)}; best = max(zs.values())
    RESULTS["bayes_factors"] = {f"n={n}": {"2dlnZ_vs_best": 2*(zs[n]-best)} for n in zs}
    print("  Bayes factors (Npart^1/3):", {f"n={n}": round(2*(zs[n]-best), 1) for n in zs})

# %% [markdown]
# ## Cell 7B — Coverage / closure calibration test (full-Hessian marginal $\sigma_n$)

# %%
if RUN_COVERAGE:
    GC = GEO["Npart13"]; fs = {s: (SQRTS[s]/5.02)**0.31 for s in ORDER}; truth = dict(k=2.0, n=2.0, b=0.3)
    MU = {nm: raa_model(FIT[nm]["pt"], 1.0, GC[nm], truth["k"]*fs[nm], truth["n"], truth["b"], a_of(FIT[nm]["pt"], SQRTS[nm]), False) for nm in ORDER}
    def nll_s(th, data):
        k, n, b = th
        if not (0 < k < 12 and 0 <= n <= 4.5 and -0.5 <= b <= 1.5): return 1e12
        v = 0.5*((n-2)/1.5)**2
        for nm in ORDER:
            dpt = k*fs[nm]*GC[nm]**n*np.power(FIT[nm]["pt"], b); r = data[nm]-(FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**a_of(FIT[nm]["pt"], SQRTS[nm]); v += 0.5*r@cho_solve(CH[nm], r)
        return v
    def hess(fn, x, h=2e-3):
        n = len(x); H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                xpp = x.copy(); xpp[i] += h; xpp[j] += h; xpm = x.copy(); xpm[i] += h; xpm[j] -= h
                xmp = x.copy(); xmp[i] -= h; xmp[j] += h; xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
                H[i, j] = H[j, i] = (fn(xpp)-fn(xpm)-fn(xmp)+fn(xmm))/(4*h*h)
        return H
    h68 = h90 = ok = 0
    for i in range(250):
        rng = np.random.default_rng(20000+i); data = {nm: rng.multivariate_normal(MU[nm], FIT[nm]["cov"]) for nm in ORDER}
        r = minimize(nll_s, [2, 2, .3], args=(data,), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
        try: sn = np.sqrt(abs(np.linalg.inv(hess(lambda t: nll_s(t, data), r.x))[1, 1]))
        except Exception: continue
        if not np.isfinite(sn) or sn <= 0: continue
        ok += 1
        if abs(2.0-r.x[1]) <= sn: h68 += 1
        if abs(2.0-r.x[1]) <= 1.645*sn: h90 += 1
    RESULTS["coverage"] = {"cov68": h68/ok, "cov90": h90/ok, "trials": ok}
    print(f"  coverage 68%->{h68/ok:.2f} 90%->{h90/ok:.2f} (nominal 0.68/0.90)")

# %% [markdown]
# ## Cell 7C — Universality test (single vs broken exponent, O+O$\to$Pb+Pb)

# %%
if RUN_UNIVERSALITY:
    from dynesty import NestedSampler
    GU = GEO["Npart13"]; SMALL = {"OO", "NeNe"}; fs = {s: (SQRTS[s]/5.02)**0.31 for s in ORDER}
    def chi2u(k, nmap, b):
        s = 0.0
        for nm in ORDER:
            dpt = k*fs[nm]*GU[nm]**nmap[nm]*np.power(FIT[nm]["pt"], b); r = FIT[nm]["raa"]-(FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**a_of(FIT[nm]["pt"], SQRTS[nm]); s += r@cho_solve(CH[nm], r)
        return s
    def Zu(): 
        s = NestedSampler(lambda th: -0.5*chi2u(th[0], {nm: th[1] for nm in ORDER}, th[2]), lambda u: np.array([12*u[0], 4.5*u[1], -0.5+2*u[2]]), 3, nlive=400, rstate=np.random.default_rng(0)); s.run_nested(print_progress=False, dlogz=0.1); return s.results.logz[-1]
    def Zb():
        s = NestedSampler(lambda th: -0.5*chi2u(th[0], {nm: (th[1] if nm in SMALL else th[2]) for nm in ORDER}, th[3]), lambda u: np.array([12*u[0], 4.5*u[1], 4.5*u[2], -0.5+2*u[3]]), 4, nlive=400, rstate=np.random.default_rng(0)); s.run_nested(print_progress=False, dlogz=0.1); return s.results.logz[-1]
    zu, zb = Zu(), Zb(); RESULTS["universality"] = {"2lnB_broken_vs_universal": 2*(zb-zu)}
    print(f"  universality: 2 ln(B_broken/universal) = {2*(zb-zu):+.2f}  ({'universal' if 2*(zb-zu)<2 else 'broken'})")

# %% [markdown]
# ## Cell 7D — $\sqrt{s}$ correction & systematic budget

# %%
if RUN_SQRTS:
    n_corr = mcmc_fit(GEO["Npart13"], False, sqrts_corr=True)["nq"][1]
    n_unc  = mcmc_fit(GEO["Npart13"], False, sqrts_corr=False)["nq"][1]
    RESULTS["sqrts"] = {"n_corrected": n_corr, "n_uncorrected": n_unc, "shift": abs(n_corr-n_unc)}
    print(f"  sqrt(s): n_corr={n_corr:.2f} vs n_uncorr={n_unc:.2f} -> shift {abs(n_corr-n_unc):.3f}")
if RUN_GEOM_SCAN:
    fs = {s: (SQRTS[s]/5.02)**0.31 for s in ORDER}
    def nfit_fast(geom):
        def nll(th):
            k, n, b = th
            if not (0 < k < 12 and 0 <= n <= 4.5 and -0.5 <= b <= 1.5): return 1e12
            v = 0.5*((n-2)/1.5)**2
            for nm in ORDER:
                dpt = k*fs[nm]*geom[nm]**n*np.power(FIT[nm]["pt"], b); r = FIT[nm]["raa"]-(FIT[nm]["pt"]/(FIT[nm]["pt"]+dpt))**a_of(FIT[nm]["pt"], SQRTS[nm]); v += 0.5*r@cho_solve(CH[nm], r)
            return v
        return minimize(nll, [2, 1.8, .3], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000}).x[1]
    ns = []
    for sg in (6.4, 7.0, 7.6):
        SIGNN_bak = SIGNN
        # only Npart^1/3 ratios depend on sigma; recompute optical quickly
    # geometry-definition spread is the dominant systematic:
    defs = {"A^1/3": {s: (A_OF[s]/208)**(1/3) for s in ORDER}, "Npart^1/3": GEO["Npart13"]}
    if "L" in GEO: defs["exit-L"] = GEO["L"]
    nd = {k: nfit_fast(v) for k, v in defs.items()}
    RESULTS["geometry_definition_n"] = nd
    print("  geometry-definition exponents:", {k: round(v, 2) for k, v in nd.items()})

# %% [markdown]
# ## Cell 8A — Machine learning: probabilistic emulator + leave-one-system-out

# %%
if RUN_ML_EMULATOR:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from ngboost import NGBRegressor; from ngboost.distns import Normal
    G = GEO["Npart13"]; pts = {nm: FIT[nm]["pt"] for nm in ORDER}; raa = {nm: FIT[nm]["raa"] for nm in ORDER}
    feats = lambda nm: np.column_stack([np.log10(pts[nm]), np.full(len(pts[nm]), G[nm])])
    def gp(X, y): m = GaussianProcessRegressor(Ck(1.)*RBF([.5, .3])+WhiteKernel(1e-3), normalize_y=True, n_restarts_optimizer=3, alpha=1e-6).fit(X, y); return m
    loo = {}
    for held in ORDER:
        tr = [s for s in ORDER if s != held]; Xtr = np.vstack([feats(s) for s in tr]); ytr = np.concatenate([raa[s] for s in tr])
        m = gp(Xtr, ytr); mu, sd = m.predict(feats(held), return_std=True)
        loo[held] = {"rmse": float(np.sqrt(np.mean((mu-raa[held])**2))), "cov68": float(np.mean(np.abs(mu-raa[held]) <= sd)), "mu": mu.tolist(), "sd": sd.tolist()}
    RESULTS["ml_emulator_loso"] = {h: {k: loo[h][k] for k in ("rmse", "cov68")} for h in loo}
    print("  LOSO GP RMSE:", {h: round(loo[h]["rmse"], 3) for h in loo})
    fig, ax = plt.subplots(figsize=(7, 4.5)); held = "NeNe"
    ax.errorbar(pts[held], raa[held], yerr=np.sqrt(np.diag(FIT[held]["cov"])), fmt="o", color="k", ms=5, capsize=3, label="NeNe measured", zorder=5)
    mu = np.array(loo[held]["mu"]); sd = np.array(loo[held]["sd"]); ax.fill_between(pts[held], mu-sd, mu+sd, color="#1f77b4", alpha=.2)
    ax.plot(pts[held], mu, color="#1f77b4", lw=1.5, label="GP (trained on OO,XeXe,PbPb)")
    ax.set_xscale("log"); ax.set_xlabel("pT [GeV]"); ax.set_ylabel("R_AA"); ax.set_title("ML: predict held-out NeNe"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    savefig(fig, "ML_emulator_NeNe.png")

# %% [markdown]
# ## Cell 8B — Neural Posterior Estimation (normalizing flow) + Simulation-Based Calibration
# Trains $q(\theta|x)$ on a simulated benchmark, validates with SBC, applies to real data,
# compares to MCMC, demonstrates amortization. Skipped automatically if torch/zuko absent.

# %%
if RUN_ML_NPE_FLOW:
    import torch, zuko; torch.manual_seed(0)
    G = GEO["Npart13"]; PTd = {nm: FIT[nm]["pt"] for nm in ORDER}; NPTS = {nm: len(PTd[nm]) for nm in ORDER}; CTX = sum(NPTS.values())
    CHl = {nm: cholesky(FIT[nm]["cov"], lower=True) for nm in ORDER}; a0 = 6.337; sa = 0.20
    LO = np.array([0.5, 0.0, -0.3, a0-4*sa]); HI = np.array([6.0, 4.5, 1.0, a0+4*sa])
    def sprior(m): u = np.random.uniform(size=(m, 4)); th = LO+u*(HI-LO); th[:, 3] = np.random.normal(a0, sa, m); return th
    def simulate(th):
        k = th[:, 0:1]; n = th[:, 1:2]; b = th[:, 2:3]; a = th[:, 3:4]; cols = []
        for nm in ORDER:
            pt = PTd[nm][None, :]; mu = (pt/(pt+k*(G[nm]**n)*np.power(pt, b)))**a + np.random.standard_normal((th.shape[0], NPTS[nm]))@CHl[nm].T
            cols.append(mu)
        return np.concatenate(cols, 1)
    Ntr = 15000; th_tr = sprior(Ntr); X_tr = simulate(th_tr)
    xm, xs = X_tr.mean(0), X_tr.std(0)+1e-9; tm, ts = th_tr.mean(0), th_tr.std(0)+1e-9
    Xn = torch.tensor((X_tr-xm)/xs, dtype=torch.float32); Tn = torch.tensor((th_tr-tm)/ts, dtype=torch.float32)
    flow = zuko.flows.NSF(features=4, context=CTX, transforms=4, hidden_features=[128, 128]); opt = torch.optim.Adam(flow.parameters(), 1e-3, weight_decay=1e-5)
    idx = np.arange(Ntr)
    for ep in range(160):
        np.random.shuffle(idx)
        for s in range(0, Ntr, 512):
            b = idx[s:s+512]; loss = -flow(Xn[b]).log_prob(Tn[b]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    # SBC
    M, L = 500, 300; th_s = sprior(M); X_s = simulate(th_s)
    with torch.no_grad(): smp = flow(torch.tensor((X_s-xm)/xs, dtype=torch.float32)).sample((L,)).numpy()*ts[None, None, :]+tm[None, None, :]
    ranks = (smp < th_s[None, :, :]).sum(0); sbc = {nm: float((((np.histogram(ranks[:, d], 20, (0, L+1))[0]-M/20)**2/(M/20)).sum())) for d, nm in enumerate(["k", "n", "b", "a"])}
    x_real = np.concatenate([FIT[nm]["raa"] for nm in ORDER])
    t1 = time.time()
    with torch.no_grad(): post = flow(torch.tensor(((x_real-xm)/xs)[None, :], dtype=torch.float32)).sample((5000,)).numpy()[:, 0, :]*ts+tm
    t_amort = (time.time()-t1)*1000
    n_npe = float(np.median(post[:, 1])); ch = np.load(os.path.join(OUT, "chain_effective.npy")); n_mcmc = float(np.median(ch[:, 1]))
    RESULTS["ml_npe"] = {"n_npe": n_npe, "n_mcmc": n_mcmc, "sbc_chi2": sbc, "amortized_ms": t_amort}
    print(f"  NPE-flow n={n_npe:.2f} vs MCMC n={n_mcmc:.2f} | SBC chi2(n)={sbc['n']:.0f} (~19) | amortized {t_amort:.0f} ms")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(post[:, 1], 50, density=True, color="#d62728", alpha=.5, label=f"NPE-flow {n_npe:.2f}")
    xx = np.linspace(0, 4, 300); ax[0].plot(xx, norm.pdf(xx, n_mcmc, np.std(ch[:, 1])), color="#1f4e79", lw=2, label=f"MCMC {n_mcmc:.2f}")
    ax[0].axvline(2, ls="--", c="green"); ax[0].set_xlabel("effective n"); ax[0].set_title("NPE-flow vs MCMC"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].hist(ranks[:, 1], 20, color="#1f4e79", alpha=.8); ax[1].axhline(M/20, ls="--", c="red"); ax[1].set_title(f"SBC rank (n): chi2={sbc['n']:.0f}"); ax[1].set_xticks([])
    savefig(fig, "ML_NPE_SBC.png")

# %% [markdown]
# ## Cell 9 — Summary figures (posterior-predictive, n vs geometry)

# %%
gk0 = list(GEO)[0]; f = BAY[gk0]["effective"]["chain"]; LRr = GEO[gk0]; rng = np.random.default_rng(1); dr = f[rng.integers(0, len(f), 300)]
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for nm in ORDER:
    d = FIT[nm]; pg = np.logspace(np.log10(d["pt"].min()), np.log10(d["pt"].max()), 60)
    band = np.array([raa_model(pg, 1.0, LRr[nm], th[0]*(SQRTS[nm]/5.02)**0.31, th[1], th[2], a_of(pg, SQRTS[nm]), False) for th in dr]); c = COL[nm]
    ax.fill_between(pg, np.percentile(band, 16, 0), np.percentile(band, 84, 0), color=c, alpha=.22); ax.plot(pg, np.percentile(band, 50, 0), color=c, lw=1.2)
    ax.errorbar(d["pt"], d["raa"], yerr=np.sqrt(np.diag(d["cov"])), fmt="o", ms=4, capsize=2, color=c, label=f"{nm} (A={d['A']})")
ax.axhline(1, ls="--", lw=.7, c="grey"); ax.set_xscale("log"); ax.set_xlabel("pT [GeV]"); ax.set_ylabel("R_AA"); ax.set_ylim(0, 1.2)
ax.legend(loc="lower right", ncol=2); ax.set_title(f"4-system fit (eff n={BAY[gk0]['effective']['nq'][1]:.2f}), REAL CMS data"); ax.grid(alpha=.25, which="both")
savefig(fig, "summary_posterior_predictive.png")

# %% [markdown]
# ## Cell 10 — Save consolidated results

# %%
json.dump(RESULTS, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
print("\n=== DONE ===  results.json keys:", list(RESULTS.keys()))
print("headline effective n (Npart^1/3):", RESULTS["bayesian"]["Npart13"]["effective"]["n"][1] if "Npart13" in RESULTS["bayesian"] else "n/a")
