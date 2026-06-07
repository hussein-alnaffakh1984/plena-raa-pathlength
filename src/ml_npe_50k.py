#!/usr/bin/env python3
"""
ML UPGRADE: Neural Posterior Estimation (conditional normalizing flow, zuko)
            + Simulation-Based Calibration (SBC)  + amortization demo.
Trains q(theta|x) on a large simulated benchmark, validates with SBC rank
histograms and expected coverage, applies to real data, compares to MCMC.
"""
import os, glob, re, json, time, yaml, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import cholesky
import torch, zuko
torch.manual_seed(0); np.random.seed(0)
OUT = "/tmp/out"; os.makedirs(OUT, exist_ok=True)
NEW = "/tmp/real4/HEPData-ins3123773-v1-yaml"; OLD = "/tmp/real"; PT_MIN = 8.0
A_OF = {"OO": 16, "NeNe": 20, "XeXe": 129, "PbPb": 208}
RHO = {"OO": 0.340, "NeNe": 0.368, "XeXe": 0.966, "PbPb": 1.0}     # from Glauber (results.json)
GN = {"OO": 0.457, "NeNe": 0.492, "XeXe": 0.911, "PbPb": 1.0}      # <Npart>^1/3 ratios
META = {"pt", "raa", "system", "A"}

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
def build_cov(df, xi=4.0):
    n = len(df); C = np.zeros((n, n)); D = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    for c in comps(df):
        v = df[c].values.astype(float); k = classify(c)
        C += np.diag(v**2) if k == "u" else (np.outer(v, v) if k == "f" else np.outer(v, v)*np.exp(-D/xi))
    return C
def restrict(df):
    C = build_cov(df); i = np.where(df["pt"].values >= PT_MIN)[0]
    return dict(pt=df["pt"].values[i], raa=df["raa"].values[i], cov=C[np.ix_(i, i)])

ORDER = ["OO", "NeNe", "XeXe", "PbPb"]
DFS = {"OO": load_new("oo_raa_(coarse_pt_binning).yaml", "OO"), "NeNe": load_new("nene_raa_(coarse_pt_binning).yaml", "NeNe"),
       "PbPb": load_new("pbpb_raa_(coarse_pt_binning).yaml", "PbPb"), "XeXe": load_xexe()}
FIT = {nm: restrict(DFS[nm]) for nm in ORDER}
PT = {nm: FIT[nm]["pt"] for nm in ORDER}; CHOL = {nm: cholesky(FIT[nm]["cov"], lower=True) for nm in ORDER}
NPTS = {nm: len(PT[nm]) for nm in ORDER}; CTX = sum(NPTS.values())
print(f"context dim (R_AA points, 4 AA systems) = {CTX}")
a0, sa = 6.337, 0.20

# ---- prior & simulator (theta=[k,n,beta,a]) ----
LO = np.array([0.5, 0.0, -0.3, a0-4*sa]); HI = np.array([6.0, 4.5, 1.0, a0+4*sa])
def sample_prior(m):
    u = np.random.uniform(size=(m, 4)); th = LO + u*(HI-LO); th[:, 3] = np.random.normal(a0, sa, m); return th
def raa_clean(pt, G, k, n, b, a): return (pt/(pt + k*G**n*np.power(pt, b)))**a
def simulate(th, noise=True):
    m = th.shape[0]; k = th[:, 0:1]; n = th[:, 1:2]; b = th[:, 2:3]; a = th[:, 3:4]; cols = []
    for nm in ORDER:
        pt = PT[nm][None, :]; term = k*(GN[nm]**n)*np.power(pt, b); mu = (pt/(pt+term))**a  # (m,P) vectorized
        if noise: mu = mu + np.random.standard_normal((m, NPTS[nm])) @ CHOL[nm].T
        cols.append(mu)
    return np.concatenate(cols, axis=1)

# ---- training set ----
Ntr = 50000; print(f"simulating {Ntr} training datasets ...")
th_tr = sample_prior(Ntr); X_tr = simulate(th_tr)
xm, xs = X_tr.mean(0), X_tr.std(0)+1e-9; tm, ts = th_tr.mean(0), th_tr.std(0)+1e-9
Xn = torch.tensor((X_tr-xm)/xs, dtype=torch.float32); Tn = torch.tensor((th_tr-tm)/ts, dtype=torch.float32)

# ---- conditional normalizing flow q(theta|x) ----
flow = zuko.flows.NSF(features=4, context=CTX, transforms=4, hidden_features=[128, 128])
opt = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
print("training normalizing-flow NPE ...")
nb = 512; idx = np.arange(Ntr); t0 = time.time()
for ep in range(80):
    np.random.shuffle(idx); tot = 0
    for s in range(0, Ntr, nb):
        b = idx[s:s+nb]; loss = -flow(Xn[b]).log_prob(Tn[b]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)*len(b)
    if ep % 40 == 0 or ep == 79: print(f"  epoch {ep:3d}  -logq/dim = {tot/Ntr:.3f}")
print(f"  trained in {time.time()-t0:.0f}s")

def posterior(x_raw, nsamp=2000):
    xc = torch.tensor(((x_raw-xm)/xs)[None, :], dtype=torch.float32)
    with torch.no_grad(): s = flow(xc).sample((nsamp,)).numpy().reshape(nsamp, 4)
    return s*ts + tm

# ---- SBC: rank uniformity over M sims (BATCHED sampling) ----
print("running Simulation-Based Calibration (SBC) ...")
M, L = 80, 120; th_sbc = sample_prior(M); X_sbc = simulate(th_sbc)
Xc = torch.tensor((X_sbc-xm)/xs, dtype=torch.float32)
with torch.no_grad(): samp = flow(Xc).sample((L,)).numpy()          # (L, M, 4) standardized
samp = samp*ts[None, None, :] + tm[None, None, :]
ranks = (samp < th_sbc[None, :, :]).sum(0)                          # (M,4)
nbin = 20; names = ["kappa", "n", "beta", "a"]; sbc = {}
for d in range(4):
    h, _ = np.histogram(ranks[:, d], bins=nbin, range=(0, L+1)); exp = M/nbin
    sbc[names[d]] = {"chi2": float(((h-exp)**2/exp).sum()), "dof": nbin-1}
print("  SBC chi2 (~19 if uniform): " + ", ".join(f"{names[d]}={sbc[names[d]]['chi2']:.1f}" for d in range(4)))
levels = np.linspace(0.05, 0.95, 19)
emp_cov = [np.mean(np.abs(ranks[:, 1]/L - 0.5) <= q/2) for q in levels]

# ---- apply to REAL data ----
x_real = np.concatenate([FIT[nm]["raa"] for nm in ORDER])
t1 = time.time(); post = posterior(x_real, nsamp=5000); t_amort = time.time()-t1
n_npe = float(np.median(post[:, 1])); n_npe_lo, n_npe_hi = np.percentile(post[:, 1], [16, 84])
try:
    ch = np.load(os.path.join(OUT, "chain_Npart_effective.npy")); n_mcmc = float(np.median(ch[:, 1])); n_mcmc_sd = float(np.std(ch[:, 1]))
except Exception: n_mcmc, n_mcmc_sd = np.nan, np.nan
print(f"\nNPE-flow effective n = {n_npe:.2f} (+{n_npe_hi-n_npe:.2f}/-{n_npe-n_npe_lo:.2f})   "
      f"vs MCMC n = {n_mcmc:.2f} +/- {n_mcmc_sd:.2f}   [amortized inference: {t_amort*1000:.0f} ms]")

RES = {"context_dim": CTX, "n_train": Ntr, "sbc": sbc, "npe_n": n_npe, "npe_n_lo": float(n_npe_lo), "npe_n_hi": float(n_npe_hi),
       "mcmc_n": n_mcmc, "mcmc_n_sd": n_mcmc_sd, "amortized_ms": t_amort*1000}
json.dump(RES, open(os.path.join(OUT, "ml_npe_sbc_50k.json"), "w"), indent=1, default=float)

# ---- figures ----
fig, ax = plt.subplots(1, 4, figsize=(13, 3.0))
for d in range(4):
    ax[d].hist(ranks[:, d], bins=nbin, range=(0, L+1), color="#1f4e79", alpha=.8)
    ax[d].axhline(M/nbin, ls="--", c="red", lw=1); ax[d].set_title(f"SBC rank: {names[d]}\nchi2={sbc[names[d]]['chi2']:.0f} (~19)", fontsize=9)
    ax[d].set_xticks([])
fig.suptitle("Simulation-Based Calibration — flat ranks = calibrated posterior", fontsize=11); fig.tight_layout()
fig.savefig(os.path.join(OUT, "ML_D_SBC_ranks.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].hist(post[:, 1], bins=50, density=True, color="#d62728", alpha=.5, label=f"NPE-flow {n_npe:.2f}")
if np.isfinite(n_mcmc):
    from scipy.stats import norm; xx = np.linspace(0, 4, 300)
    ax[0].plot(xx, norm.pdf(xx, n_mcmc, n_mcmc_sd), color="#1f4e79", lw=2, label=f"MCMC {n_mcmc:.2f}")
ax[0].axvline(2, ls="--", c="green"); ax[0].set_xlabel("effective n"); ax[0].set_title("NPE-flow vs MCMC (real data)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(levels, emp_cov, "o-", color="#1f4e79", ms=3); ax[1].plot([0, 1], [0, 1], "--", c="grey")
ax[1].set_xlabel("nominal credible level"); ax[1].set_ylabel("empirical coverage (SBC)"); ax[1].set_title("Expected coverage of n (SBC)"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("/tmp/out/ML_E_NPE_posterior_50k.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved ML_D_SBC_ranks.png, ML_E_NPE_posterior.png, ml_npe_sbc_50k.json")
print("DONE")
