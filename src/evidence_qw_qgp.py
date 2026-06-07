#!/usr/bin/env python3
"""Reviewer P1 + PRC-upgrade computations:
 (A) Continuous evidence landscape 2dlnZ(n_fixed) for n in [0.5,3.5].
 (B) Quenching-weight Ansatz (log-normal P(dE) convolution) as forward-model
     variant; check n_eff stability vs the momentum-shift model.
 (C) Bayes factor for QGP (energy loss) in O+O alone: fitted model vs no-loss.
Outputs: evidence_landscape.json, qw_variant.json, qgp_oo.json + figure."""
import os,glob,json,warnings; warnings.filterwarnings("ignore")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.integrate import quad
from dynesty import NestedSampler

NEW="/tmp/real4/HEPData-ins3123773-v1-yaml"; XE=glob.glob("/tmp/real/HEPData-ins1692558*/")[0]
SQRTS={"OO":5.36,"NeNe":5.36,"XeXe":5.44,"PbPb":5.02}; ORDER=["OO","NeNe","XeXe","PbPb"]
APOLY=np.load("/tmp/a_poly.npy"); a_local=lambda pt,s:(lambda l:APOLY[0]*l*l+APOLY[1]*l+APOLY[2])(np.log(np.asarray(pt)*5.02/s))
G=json.load(open("/tmp/out/mc_glauber.json"))["mc_ratios"]["Npart13"]
import re
def parse(d):
    iv=d["independent_variables"][0]["values"]; dv=d["dependent_variables"][0]["values"]; rows=[]
    for i,val in enumerate(dv):
        v=iv[i]; pt=0.5*(float(v["low"])+float(v["high"])) if "low" in v else float(v["value"]); row={"pt":pt,"raa":float(val["value"])}
        for e in val.get("errors",[]):
            lab=(e.get("label") or "e").lower(); row[lab]=abs(float(e["symerror"])) if "symerror" in e else 0.5*(abs(float(e["asymerror"]["plus"]))+abs(float(e["asymerror"]["minus"])))
        rows.append(row)
    return rows
def loadn(fn): return parse(yaml.safe_load(open(f"{NEW}/{fn}")))
def loadxe():
    F={os.path.basename(p):open(p).read() for p in glob.glob(XE+"/*.yaml")}; best=None;span=-1
    for d in yaml.safe_load_all(F["submission.yaml"]):
        if isinstance(d,dict) and "data_file" in d:
            kw={k["name"]:k.get("values",[]) for k in d.get("keywords",[])}
            if any("RAA" in str(x).upper() for x in kw.get("observables",[])):
                dd=yaml.safe_load(F[d["data_file"]]);cen="n/a"
                for q in dd["dependent_variables"][0].get("qualifiers",[]):
                    if "CENTRALITY" in str(q.get("name","")).upper(): cen=str(q.get("value",""))
                m=re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",cen.lower());sp=float(m[0][1])-float(m[0][0]) if m else 100
                if sp>span: span=sp;best=dd
    return parse(best)
RAW={"OO":loadn("oo_raa_(coarse_pt_binning).yaml"),"NeNe":loadn("nene_raa_(coarse_pt_binning).yaml"),
     "PbPb":loadn("pbpb_raa_(coarse_pt_binning).yaml"),"XeXe":loadxe()}
def cls(n):
    n=n.lower(); return "u" if "stat" in n else ("f" if any(k in n for k in("taa","lumi","norm","global")) else "p")
def build(nm,xi=4.0):
    rows=RAW[nm];pts=np.array([r["pt"] for r in rows]);raa=np.array([r["raa"] for r in rows])
    comps=set().union(*[set(r) for r in rows])-{"pt","raa"};n=len(rows);C=np.zeros((n,n));D=np.abs(np.subtract.outer(np.arange(n),np.arange(n)))
    for c in comps:
        v=np.array([r.get(c,0.) for r in rows]);k=cls(c);C+=np.diag(v**2) if k=="u" else (np.outer(v,v) if k=="f" else np.outer(v,v)*np.exp(-D/xi))
    m=pts>=8.0;idx=np.where(m)[0];return pts[m],raa[m],C[np.ix_(idx,idx)]
DAT={s:build(s) for s in ORDER}; CH={s:cho_factor(DAT[s][2]) for s in ORDER}; fs={s:(SQRTS[s]/5.02)**0.31 for s in ORDER}

# ---------- shift-model likelihood ----------
def raa_shift(pt,s,k,n,b): return (pt/(pt+k*fs[s]*G[s]**n*np.power(pt,b)))**a_local(pt,SQRTS[s])
def chi2_shift(k,n,b,systems=ORDER):
    t=0.
    for s in systems:
        pt,raa,_=DAT[s]; r=raa-raa_shift(pt,s,k,n,b); t+=r@cho_solve(CH[s],r)
    return t

# ===== (A) Evidence landscape: dynesty evidence at fixed n over a grid =====
print("== (A) Evidence landscape 2dlnZ(n) ==")
def lnZ_fixed_n(nfix):
    def ll(th):
        k,b=th; return -0.5*chi2_shift(k,nfix,b)
    def pt_(u): return np.array([12*u[0],-0.5+2*u[1]])
    s=NestedSampler(ll,pt_,2,nlive=350,rstate=np.random.default_rng(0))
    s.run_nested(print_progress=False,dlogz=0.2); return float(s.results.logz[-1])
ngrid=np.arange(0.5,3.51,0.25)
lnz={float(n):lnZ_fixed_n(float(n)) for n in ngrid}
best=max(lnz.values()); land={n:2*(lnz[n]-best) for n in lnz}
npeak=max(lnz,key=lnz.get)
print("  peak at n =",npeak,"; 2dlnZ at n=1:",round(land.get(1.0,np.nan),1),"n=2:",round(land.get(2.0,np.nan),1),"n=3:",round(land.get(3.0,np.nan),1))
json.dump({"n":list(land.keys()),"twodlnz":list(land.values()),"peak":npeak},open("/tmp/out/evidence_landscape.json","w"),indent=1)

# ===== (B) Quenching-weight Ansatz: log-normal P(dE), R_AA = <(pt/(pt+dE))^a> =====
print("\n== (B) Quenching-weight (log-normal P(dE)) variant ==")
# mean energy loss <dE> = k * fs * G^n * pt^b ; lognormal with shape sigma_w (width of P(dE))
SIGMA_W=0.5  # lognormal sigma (sigma_L/L ~ 0.25 -> dE width O(0.5))
def raa_qw(pt,s,k,n,b,nq=12):
    a=a_local(pt,SQRTS[s]); mean=k*fs[s]*G[s]**n*np.power(pt,b)
    # lognormal samples of dE with given mean and shape SIGMA_W
    mu=np.log(np.maximum(mean,1e-6))-0.5*SIGMA_W**2
    # Gauss-Hermite over ln(dE)
    x,w=np.polynomial.hermite_e.hermegauss(nq); w=w/np.sqrt(2*np.pi)
    out=np.zeros_like(pt,dtype=float)
    for xi_,wi in zip(x,w):
        dE=np.exp(mu+SIGMA_W*xi_); out+=wi*(pt/(pt+dE))**a
    return out
def chi2_qw(k,n,b):
    t=0.
    for s in ORDER:
        pt,raa,_=DAT[s]; r=raa-raa_qw(pt,s,k,n,b); t+=r@cho_solve(CH[s],r)
    return t
def fit(chi2fn,x0=(2,1.8,.3)):
    r=minimize(lambda th:0.5*chi2fn(*th)+0.5*((th[1]-2)/1.5)**2 if (0<th[0]<12 and 0<=th[1]<=4.5 and -0.5<=th[2]<=1.5) else 1e12,
               x0,method="Nelder-Mead",options={"xatol":1e-4,"fatol":1e-7,"maxiter":8000})
    return r.x
ks,ns,bs=fit(chi2_shift); kq,nq_,bq=fit(chi2_qw)
print(f"  shift-model   n_eff={ns:.3f}  (chi2/dof={chi2_shift(ks,ns,bs)/37:.2f})")
print(f"  quench-weight n_eff={nq_:.3f}  (chi2/dof={chi2_qw(kq,nq_,bq)/37:.2f})  sigma_w={SIGMA_W}")
print(f"  -> n shift from forward-model form = {abs(nq_-ns):.3f}")
json.dump({"shift_n":float(ns),"qw_n":float(nq_),"sigma_w":SIGMA_W,"dn_form":float(abs(nq_-ns))},open("/tmp/out/qw_variant.json","w"),indent=1)

# ===== (C) Bayes factor for QGP (energy loss) in O+O alone =====
print("\n== (C) QGP-in-OO Bayes factor: loss vs no-loss ==")
# Model M1 (loss): fit (k,b) at n=2 to OO only.  Model M0 (no loss): R_AA=1 (kappa=0).
def lnZ_oo_loss():
    def ll(th):
        k,b=th; pt,raa,_=DAT["OO"]; r=raa-raa_shift(pt,"OO",k,2.0,b); return -0.5*r@cho_solve(CH["OO"],r)
    s=NestedSampler(ll,lambda u:np.array([12*u[0],-0.5+2*u[1]]),2,nlive=400,rstate=np.random.default_rng(1))
    s.run_nested(print_progress=False,dlogz=0.1); return float(s.results.logz[-1])
def lnZ_oo_noloss():
    pt,raa,_=DAT["OO"]; r=raa-1.0  # R_AA=1 everywhere
    return float(-0.5*r@cho_solve(CH["OO"],r))   # no free params (point model)
zl=lnZ_oo_loss(); z0=lnZ_oo_noloss(); b_oo=2*(zl-z0)
print(f"  OO: 2dlnZ(loss vs no-loss) = {b_oo:.1f}  (>10 = decisive evidence for energy loss / QGP)")
json.dump({"twodlnz_loss_vs_noloss":float(b_oo),"lnz_loss":zl,"lnz_noloss":z0},open("/tmp/out/qgp_oo.json","w"),indent=1)

# ---------- figure: evidence landscape ----------
fig,ax=plt.subplots(figsize=(7,4.6))
ns_=sorted(land); ys=[land[n] for n in ns_]
ax.plot(ns_,ys,"-o",color="#1f4e79",ms=4,lw=1.8)
ax.axhline(0,ls=":",c="grey"); ax.axhline(-10,ls="--",c="#c0504d",lw=1,label="Jeffreys decisive ($2\\Delta\\ln Z=-10$)")
for nv,lab in [(1,"collisional"),(2,"radiative"),(3,"strong coupling")]:
    ax.axvline(nv,ls=":",c="green" if nv==2 else "grey",alpha=.6)
    ax.text(nv,3,lab,rotation=90,va="bottom",ha="right",fontsize=8,color="green" if nv==2 else "grey")
ax.set_xlabel("fixed path-length exponent $n$"); ax.set_ylabel("$2\\Delta\\ln\\mathcal{Z}$ (rel. to peak)")
ax.set_title("Continuous evidence landscape (dynesty)"); ax.legend(fontsize=8,loc="lower center"); ax.grid(alpha=.25)
ax.set_ylim(min(ys)-3,6)
fig.tight_layout(); fig.savefig("/tmp/out/PHYS_evidence_landscape.png",dpi=150,bbox_inches="tight")
print("\nsaved evidence_landscape.json, qw_variant.json, qgp_oo.json, PHYS_evidence_landscape.png")
