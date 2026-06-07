#!/usr/bin/env python3
"""Quantitative comparison with JETSCAPE (referee point 6).
Maps the extracted radiative energy-loss scale to an effective qhat/T^3 via the
BDMPS-Z mean-energy-loss relation, and overlays the Pb+Pb R_AA implied by the
JETSCAPE qhat/T^3 = 2-4 band on the data and our fit.
All assumptions are explicit; this is an order-of-magnitude consistency check,
not a transport calculation."""
import os,glob,json,warnings; warnings.filterwarnings("ignore")
import numpy as np, yaml, emcee
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

NEW="/tmp/real4/HEPData-ins3123773-v1-yaml"; XE=glob.glob("/tmp/real/HEPData-ins1692558*/")[0]
A_OF={"OO":16,"NeNe":20,"XeXe":129,"PbPb":208}; SQRTS={"OO":5.36,"NeNe":5.36,"XeXe":5.44,"PbPb":5.02}; ORDER=["OO","NeNe","XeXe","PbPb"]
APOLY=np.load("/tmp/a_poly.npy"); a_local=lambda pt,s:(lambda l:APOLY[0]*l*l+APOLY[1]*l+APOLY[2])(np.log(np.asarray(pt)*5.02/s))
G=json.load(open("/tmp/out/mc_glauber.json"))["mc_ratios"]["Npart13"]
Labs={s:json.load(open("/tmp/out/mc_glauber.json"))["mc"][s]["L"] for s in ORDER}
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
D={s:build(s) for s in ORDER};CH={s:cho_factor(D[s][2]) for s in ORDER};fs={s:(SQRTS[s]/5.02)**0.31 for s in ORDER}

# ---- refit (kappa,n,beta) for Npart-MC, effective ----
def logp(th):
    k,n,b=th
    if not(0<k<12 and 0<=n<=4.5 and -0.5<=b<=1.5): return -np.inf
    ll=-0.5*((n-2)/1.5)**2
    for s in ORDER:
        pt,raa,_=D[s];dpt=k*fs[s]*G[s]**n*np.power(pt,b);r=raa-(pt/(pt+dpt))**a_local(pt,SQRTS[s]);ll+=-0.5*r@cho_solve(CH[s],r)
    return ll
rng=np.random.default_rng(0); p0=np.array([2.,1.8,.3])+0.02*rng.standard_normal((48,3))
sm=emcee.EnsembleSampler(48,3,logp); sm.run_mcmc(p0,6000,progress=False)
ch=sm.get_chain(discard=2400,thin=10,flat=True)
kap,nn,bet=np.median(ch,0); print(f"refit Npart-MC: kappa={kap:.3f} n={nn:.2f} beta={bet:.2f}")

# ---- map our energy-loss scale to qhat/T^3 (BDMPS) ----
HBARC=0.19733  # GeV*fm
def dpt_pbpb(pt,k=kap,b=bet): return k*1.0*pt**b      # G_PbPb=1, fs=1
def qhat_over_T3(dE, L, T, alphas, CR):
    # BDMPS mean radiative loss: <dE> = (alphas*CR/4) qhat L^2  -> qhat [GeV^2/fm]
    qhat = 4.0*dE/(alphas*CR*L*L)                     # GeV^2/fm
    return qhat*HBARC/T**3                             # dimensionless
pt_ref=10.0; dE=dpt_pbpb(pt_ref)                       # GeV (approx energy loss at 10 GeV)
print(f"PbPb <dpT> at {pt_ref} GeV = {dE:.2f} GeV  (fractional {dE/pt_ref:.2f})")
# central + ranges
grid=[]
for L in (Labs["PbPb"], 5.0, 6.0):
    for T in (0.30,0.35,0.40):
        for CR in (4/3,3.0):
            grid.append(qhat_over_T3(dE,L,T,0.3,CR))
grid=np.array(grid)
print(f"effective qhat/T^3: median={np.median(grid):.1f} range=[{grid.min():.1f},{grid.max():.1f}]  (JETSCAPE 90% CR: 2-4)")
# propagate kappa uncertainty at central (L=MC,T=0.35,CR=3)
dE_post=np.array([dpt_pbpb(pt_ref,k=c[0],b=c[2]) for c in ch[::5]])
q_post=qhat_over_T3(dE_post,Labs["PbPb"],0.35,0.3,3.0)
print(f"qhat/T^3 at central assumptions: {np.median(q_post):.1f} (+{np.percentile(q_post,84)-np.median(q_post):.1f}/-{np.median(q_post)-np.percentile(q_post,16):.1f})")

# ---- inverse: R_AA(PbPb) band for JETSCAPE qhat/T^3 in [2,4] ----
def kappa_from_q(qT3, L, T, alphas, CR, b):
    qhat=qT3*T**3/HBARC                                # GeV^2/fm
    dE=(alphas*CR/4)*qhat*L*L                           # GeV at ref
    return dE/pt_ref**b                                 # kappa s.t. kappa*pt^b = dE at pt_ref
def raa_pbpb(pt,k,b): return (pt/(pt+k*pt**b))**a_local(pt,5.02)
pg=np.logspace(np.log10(8),np.log10(100),60)
# JETSCAPE band: vary qT3 in [2,4] and L in [MC,6], T in [0.3,0.4], CR in {4/3,3}
curves=[]
for qT3 in (2,3,4):
    for L in (Labs["PbPb"],5.0,6.0):
        for T in (0.30,0.35,0.40):
            for CR in (4/3,3.0):
                kj=kappa_from_q(qT3,L,T,0.3,CR,bet); curves.append(raa_pbpb(pg,kj,bet))
curves=np.array(curves); lo=np.percentile(curves,10,0); hi=np.percentile(curves,90,0)
# our fit band
ourfit=raa_pbpb(pg,kap,bet)
pt,raa,cov=D["PbPb"]; err=np.sqrt(np.diag(cov))
fig,ax=plt.subplots(figsize=(7.2,4.8))
ax.fill_between(pg,lo,hi,color="#b7791f",alpha=.25,label="JETSCAPE $\\hat{q}/T^3{=}2{-}4$ (BDMPS map)")
ax.plot(pg,ourfit,color="#1f4e79",lw=2,label=f"This work (radiative fit, $n={nn:.2f}$)")
ax.errorbar(pt,raa,yerr=err,fmt="o",color="k",ms=5,capsize=3,label="CMS Pb+Pb")
ax.axhline(1,ls="--",lw=.7,c="grey"); ax.set_xscale("log"); ax.set_xlabel("$p_T$ [GeV]"); ax.set_ylabel("$R_{AA}$")
ax.set_ylim(0,1.15); ax.set_title("Consistency with JETSCAPE $\\hat{q}$ (Pb+Pb)"); ax.legend(fontsize=8.5,loc="lower right"); ax.grid(alpha=.25,which="both")
fig.tight_layout(); fig.savefig("/tmp/out/PHYS_jetscape_comparison.png",dpi=150,bbox_inches="tight")
print("saved PHYS_jetscape_comparison.png")
json.dump(dict(kappa=float(kap),n=float(nn),beta=float(bet),
               dpt_pbpb_10=float(dE),
               qhatT3_median=float(np.median(grid)),qhatT3_min=float(grid.min()),qhatT3_max=float(grid.max()),
               qhatT3_central=float(np.median(q_post)),
               qhatT3_central_lo=float(np.median(q_post)-np.percentile(q_post,16)),
               qhatT3_central_hi=float(np.percentile(q_post,84)-np.median(q_post)),
               jetscape_ref="qhat/T^3 = 2-4 (90% CR, MATTER+LBT, PRC 104 024905)"),
          open("/tmp/out/jetscape_comparison.json","w"),indent=1)
print("saved jetscape_comparison.json")
