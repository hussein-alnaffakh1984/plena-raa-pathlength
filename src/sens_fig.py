#!/usr/bin/env python3
import os,glob,json,warnings; warnings.filterwarnings("ignore")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

S=json.load(open("/tmp/out/sensitivity.json"))
PROXIES=["A^1/3","Npart-opt","Npart-MC","exitL-opt","exitL-MC"]; XIS=[2,4,6,8]
# n_eff at nominal (baseline=local, prior=gauss, sqrts=True) from records
rec=S["neff_records"]
def neff_at(proxy,xi):
    for r in rec:
        if r["proxy"]==proxy and r["xi"]==xi and r["baseline"]=="local" and r["prior"]=="gauss" and r["sqrts"]:
            return r["n_eff"]
    return np.nan
Mneff=np.array([[neff_at(p,x) for x in XIS] for p in PROXIES])
bg=S["bayes_grid"]
Mn1=np.array([[bg[f"{p}|xi={x}"]["n=1"] for x in XIS] for p in PROXIES])
Mn3=np.array([[bg[f"{p}|xi={x}"]["n=3"] for x in XIS] for p in PROXIES])

fig,ax=plt.subplots(1,3,figsize=(13,4.2))
labels=["A$^{1/3}$","N$_{part}$ (opt)","N$_{part}$ (MC)","exit-L (opt)","exit-L (MC)"]
def heat(a,M,title,cmap,vmin,vmax,fmt="%.2f",cbar_label=""):
    im=a.imshow(M,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    a.set_xticks(range(len(XIS))); a.set_xticklabels([f"$\\xi$={x}" for x in XIS])
    a.set_yticks(range(len(PROXIES))); a.set_yticklabels(labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j,i,fmt%M[i,j],ha="center",va="center",fontsize=8.5,
                   color="white" if (cmap=="viridis" and M[i,j]<vmin+0.4*(vmax-vmin)) else "black")
    a.set_title(title,fontsize=10.5); plt.colorbar(im,ax=a,fraction=0.046,pad=0.04,label=cbar_label)
heat(ax[0],Mneff,"Effective exponent $n_{\\rm eff}$","viridis",1.5,3.3,cbar_label="$n_{\\rm eff}$")
heat(ax[1],Mn1,"Model selection: $2\\Delta\\ln Z$ ($n{=}1$)","RdBu",-80,10,"%.0f")
heat(ax[2],Mn3,"Model selection: $2\\Delta\\ln Z$ ($n{=}3$)","RdBu",-80,10,"%.0f")
for a in ax[1:]:
    # mark cells where that fixed-n is FAVORED (>=0) with a ring
    pass
fig.suptitle("Sensitivity of the path-length extraction (geometry proxy $\\times$ covariance length $\\xi$)",
             fontsize=12,y=1.02)
fig.text(0.5,-0.06,"Radiative ($n{=}2$) favoured wherever $2\\Delta\\ln Z<0$ for both $n{=}1$ and $n{=}3$. "
         "The only exception is the optical exit-$L$ proxy (independently shown biased by the MC Glauber).",
         ha="center",fontsize=8.5,style="italic")
fig.tight_layout()
fig.savefig("/tmp/out/PHYS_sensitivity_heatmap.png",dpi=150,bbox_inches="tight")
print("saved PHYS_sensitivity_heatmap.png")

# -------- dynesty validation on reference case (Npart-MC, xi=4) --------
print("== dynesty validation (Npart-MC, xi=4) vs Laplace ==")
NEW="/tmp/real4/HEPData-ins3123773-v1-yaml"; XE=glob.glob("/tmp/real/HEPData-ins1692558*/")[0]
A_OF={"OO":16,"NeNe":20,"XeXe":129,"PbPb":208}; SQRTS={"OO":5.36,"NeNe":5.36,"XeXe":5.44,"PbPb":5.02}; ORDER=["OO","NeNe","XeXe","PbPb"]
APOLY=np.load("/tmp/a_poly.npy"); a_local=lambda pt,s:(lambda l:APOLY[0]*l*l+APOLY[1]*l+APOLY[2])(np.log(np.asarray(pt)*5.02/s))
G=json.load(open("/tmp/out/mc_glauber.json"))["mc_ratios"]["Npart13"]
def parse(d):
    iv=d["independent_variables"][0]["values"]; dv=d["dependent_variables"][0]["values"]; rows=[]
    for i,val in enumerate(dv):
        v=iv[i]; pt=0.5*(float(v["low"])+float(v["high"])) if "low" in v else float(v["value"]); row={"pt":pt,"raa":float(val["value"])}
        for e in val.get("errors",[]):
            lab=(e.get("label") or "e").lower(); row[lab]=abs(float(e["symerror"])) if "symerror" in e else 0.5*(abs(float(e["asymerror"]["plus"]))+abs(float(e["asymerror"]["minus"])))
        rows.append(row)
    return rows
def loadn(fn): return parse(yaml.safe_load(open(f"{NEW}/{fn}")))
import re
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
D={s:build(s) for s in ORDER};CH={s:cho_factor(D[s][2]) for s in ORDER};fs={s:(SQRTS[s]/5.02)**0.31 for s in ORDER}
def ll_fixed(th,nf):
    k,b=th;t=0.
    for s in ORDER:
        pt,raa,_=D[s];dpt=k*fs[s]*G[s]**nf*np.power(pt,b);r=raa-(pt/(pt+dpt))**a_local(pt,SQRTS[s]);t+=-0.5*r@cho_solve(CH[s],r)
    return t
from dynesty import NestedSampler
def Z(nf):
    s=NestedSampler(lambda th:ll_fixed(th,nf),lambda u:np.array([12*u[0],-0.5+2*u[1]]),2,nlive=400,rstate=np.random.default_rng(0))
    s.run_nested(print_progress=False,dlogz=0.1);return float(s.results.logz[-1])
z={n:Z(float(n)) for n in (1,2,3)};best=max(z.values())
dyn={n:round(2*(z[n]-best),1) for n in z}
lap={n:S["bayes_grid"]["Npart-MC|xi=4"][f"n={n}"] for n in (1,2,3)}
print("  dynesty:",dyn); print("  Laplace:",lap)
json.dump({"dynesty":dyn,"laplace":lap},open("/tmp/out/sensitivity_dynesty_check.json","w"),indent=1)
print("saved sensitivity_dynesty_check.json")
