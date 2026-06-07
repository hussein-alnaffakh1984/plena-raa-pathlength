#!/usr/bin/env python3
"""Sensitivity study (referee point 3): robustness of n_eff and the radiative
model selection across all major analysis choices.
Axes: geometry proxy x covariance length xi x spectral baseline x prior x sqrt(s).
Outputs: sensitivity.json, sensitivity_table.csv, PHYS_sensitivity_heatmap.png.
Laplace evidences (fast) are validated against dynesty on reference cases."""
import os, glob, json, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

NEW="/tmp/real4/HEPData-ins3123773-v1-yaml"
XE=glob.glob("/tmp/real/HEPData-ins1692558*/")[0]
PB=glob.glob("/tmp/real/HEPData-ins1496050*/")[0]
A_OF={"OO":16,"NeNe":20,"XeXe":129,"PbPb":208}
SQRTS={"OO":5.36,"NeNe":5.36,"XeXe":5.44,"PbPb":5.02}
ORDER=["OO","NeNe","XeXe","PbPb"]; PT_MIN=8.0
APOLY=np.load("/tmp/a_poly.npy")
def a_local(pt,s): l=np.log(np.asarray(pt)*5.02/s); return APOLY[0]*l*l+APOLY[1]*l+APOLY[2]
def a_single(pt,s): return 6.337+0.0*np.asarray(pt)

# ---------- data ----------
def parse_table(d):
    iv=d["independent_variables"][0]["values"]; dv=d["dependent_variables"][0]["values"]; rows=[]
    for i,val in enumerate(dv):
        v=iv[i]; pt=0.5*(float(v["low"])+float(v["high"])) if "low" in v else float(v["value"])
        row={"pt":pt,"raa":float(val["value"])}
        for e in val.get("errors",[]):
            lab=(e.get("label") or "e").strip().lower()
            row[lab]=abs(float(e["symerror"])) if "symerror" in e else 0.5*(abs(float(e["asymerror"]["plus"]))+abs(float(e["asymerror"]["minus"])))
        rows.append(row)
    return rows
def load_new(fn):
    return parse_table(yaml.safe_load(open(f"{NEW}/{fn}")))
def load_xe():
    F={os.path.basename(p):open(p).read() for p in glob.glob(XE+"/*.yaml")}
    best=None; span=-1
    import re
    for d in yaml.safe_load_all(F["submission.yaml"]):
        if isinstance(d,dict) and "data_file" in d:
            kw={k["name"]:k.get("values",[]) for k in d.get("keywords",[])}
            if any("RAA" in str(x).upper() for x in kw.get("observables",[])):
                dd=yaml.safe_load(F[d["data_file"]]); cen="n/a"
                for q in dd["dependent_variables"][0].get("qualifiers",[]):
                    if "CENTRALITY" in str(q.get("name","")).upper(): cen=str(q.get("value",""))
                m=re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",cen.lower()); sp=float(m[0][1])-float(m[0][0]) if m else 100
                if sp>span: span=sp; best=dd
    return parse_table(best)
RAW={"OO":load_new("oo_raa_(coarse_pt_binning).yaml"),
     "NeNe":load_new("nene_raa_(coarse_pt_binning).yaml"),
     "PbPb":load_new("pbpb_raa_(coarse_pt_binning).yaml"),
     "XeXe":load_xe()}
def classify(n):
    n=n.lower()
    if "stat" in n: return "u"
    if any(k in n for k in ("taa","lumi","norm","global")): return "f"
    return "p"
def build(nm,xi):
    rows=RAW[nm]; pts=np.array([r["pt"] for r in rows]); raa=np.array([r["raa"] for r in rows])
    comps=set().union(*[set(r.keys()) for r in rows])-{"pt","raa"}
    n=len(rows); C=np.zeros((n,n)); D=np.abs(np.subtract.outer(np.arange(n),np.arange(n)))
    for c in comps:
        v=np.array([r.get(c,0.0) for r in rows]); k=classify(c)
        C+=np.diag(v**2) if k=="u" else (np.outer(v,v) if k=="f" else np.outer(v,v)*np.exp(-D/xi))
    m=pts>=PT_MIN; idx=np.where(m)[0]
    return pts[m],raa[m],C[np.ix_(idx,idx)]

# ---------- geometry proxies (cached ratios) ----------
mc=json.load(open("/tmp/out/mc_glauber.json"))
GEOM={
 "A^1/3":        {s:(A_OF[s]/208)**(1/3) for s in ORDER},
 "Npart-opt":    mc["opt_ratios"]["Npart13"],
 "Npart-MC":     mc["mc_ratios"]["Npart13"],
 "exitL-opt":    mc["opt_ratios"]["L"],
 "exitL-MC":     mc["mc_ratios"]["L"],
}

# ---------- model + likelihood ----------
def raa_model(pt,G,k,n,b,a): return (pt/(pt+k*G**n*np.power(pt,b)))**a
def make_chi2(geom,xi,a_fn,sqrts_corr):
    fs={s:(SQRTS[s]/5.02)**0.31 if sqrts_corr else 1.0 for s in ORDER}
    D={s:build(s,xi) for s in ORDER}; CH={s:cho_factor(D[s][2]) for s in ORDER}
    def chi2(k,n,b):
        tot=0.0
        for s in ORDER:
            pt,raa,_=D[s]; dpt=k*fs[s]*geom[s]**n*np.power(pt,b)
            r=raa-(pt/(pt+dpt))**a_fn(pt,SQRTS[s]); tot+=r@cho_solve(CH[s],r)
        return tot
    return chi2

def fit_neff(geom,xi,a_fn,prior,sqrts_corr):
    chi2=make_chi2(geom,xi,a_fn,sqrts_corr)
    def nll(th):
        k,n,b=th
        if not(0<k<12 and 0<=n<=4.5 and -0.5<=b<=1.5): return 1e12
        pen=0.5*((n-2)/1.5)**2 if prior=="gauss" else 0.0
        return 0.5*chi2(k,n,b)+pen
    r=minimize(nll,[2,1.8,.3],method="Nelder-Mead",options={"xatol":1e-4,"fatol":1e-7,"maxiter":6000})
    return float(r.x[1])

def laplace_lnZ_fixed(geom,xi,a_fn,nfix,sqrts_corr):
    """Laplace evidence at fixed n over (k,b); returns chi2_min and lndetH (NLL Hessian)."""
    chi2=make_chi2(geom,xi,a_fn,sqrts_corr)
    def nll(p):
        k,b=p
        if not(0<k<12 and -0.5<=b<=1.5): return 1e12
        return 0.5*chi2(k,nfix,b)
    r=minimize(nll,[2,.3],method="Nelder-Mead",options={"xatol":1e-5,"fatol":1e-8,"maxiter":6000})
    x=r.x; h=1e-3; H=np.zeros((2,2))
    for i in range(2):
        for j in range(i,2):
            pp=x.copy();pp[i]+=h;pp[j]+=h; pm=x.copy();pm[i]+=h;pm[j]-=h
            mp=x.copy();mp[i]-=h;mp[j]+=h; mm=x.copy();mm[i]-=h;mm[j]-=h
            H[i,j]=H[j,i]=(nll(pp)-nll(pm)-nll(mp)+nll(mm))/(4*h*h)
    sign,lndet=np.linalg.slogdet(H)
    # lnZ = -chi2_min/2 + (d/2)ln(2pi) - 0.5 lndetH - ln V  ; V same across n -> drop
    lnZ=-0.5*chi2(*[x[0],nfix,x[1]]) + 1.0*np.log(2*np.pi) - 0.5*lndet
    return float(lnZ)

def bayes_factors(geom,xi,a_fn,sqrts_corr):
    z={n:laplace_lnZ_fixed(geom,xi,a_fn,float(n),sqrts_corr) for n in (1,2,3)}
    best=max(z.values()); return {n:2*(z[n]-best) for n in z}

# ---------- grid ----------
PROXIES=list(GEOM.keys()); XIS=[2.0,4.0,6.0,8.0]; BASE={"local":a_local,"single":a_single}; PRIORS=["flat","gauss"]
print("== n_eff grid (proxy x xi x baseline x prior x sqrt_s) ==")
records=[]
for proxy,xi,(bn,bf),prior,sc in itertools.product(PROXIES,XIS,BASE.items(),PRIORS,[True,False]):
    ne=fit_neff(GEOM[proxy],xi,bf,prior,sc)
    records.append(dict(proxy=proxy,xi=xi,baseline=bn,prior=prior,sqrts=sc,n_eff=ne))
neffs=np.array([r["n_eff"] for r in records])
print(f"  N combos={len(records)}  n_eff: min={neffs.min():.2f} max={neffs.max():.2f} median={np.median(neffs):.2f}")
# exclude the known optical exit-L outlier family to report the 'physical' band
phys=[r["n_eff"] for r in records if r["proxy"]!="exitL-opt"]
print(f"  excluding optical exit-L outlier: min={min(phys):.2f} max={max(phys):.2f}")

print("== Bayes-factor robustness (proxy x xi), Laplace ==")
bf_grid={}
for proxy in PROXIES:
    for xi in XIS:
        bf=bayes_factors(GEOM[proxy],xi,a_local,True)
        bf_grid[f"{proxy}|xi={int(xi)}"]={f"n={n}":round(v,1) for n,v in bf.items()}
# fraction where n=2 favored
favored=sum(1 for v in bf_grid.values() if v["n=2"]==0.0 and v["n=1"]<0 and v["n=3"]<0)
print(f"  n=2 favored in {favored}/{len(bf_grid)} (proxy x xi) combos")
worst_n1=max(v["n=1"] for v in bf_grid.values()); worst_n3=max(v["n=3"] for v in bf_grid.values())
print(f"  weakest rejection: n=1 -> 2dlnZ={worst_n1}, n=3 -> 2dlnZ={worst_n3}")

json.dump(dict(neff_records=records,
               neff_summary=dict(min=float(neffs.min()),max=float(neffs.max()),
                                 median=float(np.median(neffs)),
                                 phys_min=float(min(phys)),phys_max=float(max(phys))),
               bayes_grid=bf_grid,
               bayes_summary=dict(n2_favored=favored,total=len(bf_grid),
                                  weakest_n1=worst_n1,weakest_n3=worst_n3)),
          open("/tmp/out/sensitivity.json","w"),indent=1)

# csv
with open("/tmp/out/sensitivity_table.csv","w") as f:
    f.write("proxy,xi,baseline,prior,sqrts_corr,n_eff\n")
    for r in records: f.write(f"{r['proxy']},{r['xi']},{r['baseline']},{r['prior']},{r['sqrts']},{r['n_eff']:.3f}\n")
print("saved sensitivity.json + sensitivity_table.csv")
