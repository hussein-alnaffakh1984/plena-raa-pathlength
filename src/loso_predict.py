#!/usr/bin/env python3
"""Referee-strengthening additions (physics, not cosmetic):
 (1) Leave-one-system-out PHYSICS stability test (drop each system, refit n).
 (2) Predictions for future systems Ar+Ar (A=40), Kr+Kr (A=84) with calibrated
     uncertainty bands, from real MC-Glauber geometry + the posterior.
 (3) Posterior-predictive p-value (PPC).
Outputs: loso_predictions.json, PHYS_loso_predictions.png."""
import os,glob,json,warnings; warnings.filterwarnings("ignore")
import numpy as np, yaml, emcee
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

NEW="/tmp/real4/HEPData-ins3123773-v1-yaml"; XE=glob.glob("/tmp/real/HEPData-ins1692558*/")[0]
SQRTS={"OO":5.36,"NeNe":5.36,"XeXe":5.44,"PbPb":5.02}; ORDER=["OO","NeNe","XeXe","PbPb"]
APOLY=np.load("/tmp/a_poly.npy"); a_local=lambda pt,s:(lambda l:APOLY[0]*l*l+APOLY[1]*l+APOLY[2])(np.log(np.asarray(pt)*5.02/s))

# ---------------- MC-Glauber (incl. Ar, Kr) ----------------
# Woods-Saxon (R[fm], a[fm], A): light/medium from electron-scattering 2pF compilations
WS={"OO":(2.608,0.513,16),"NeNe":(2.80,0.55,20),"ArAr":(3.53,0.542,40),
    "KrKr":(4.39,0.54,84),"XeXe":(5.40,0.59,129),"PbPb":(6.62,0.546,208)}
SIGNN=7.0; DMIN=0.4
def sample_nuc(R,a,A):
    rmax=R+8*a; pos=np.empty((A,3)); got=0; tries=0
    while got<A and tries<300000:
        r=rmax*np.cbrt(np.random.uniform(size=4*(A-got)))
        r=r[np.random.uniform(size=len(r))<1.0/(1.0+np.exp((r-R)/a))]
        if len(r)==0: tries+=1; continue
        ct=np.random.uniform(-1,1,len(r)); ph=np.random.uniform(0,2*np.pi,len(r)); st=np.sqrt(1-ct*ct)
        for c in np.column_stack([r*st*np.cos(ph),r*st*np.sin(ph),r*ct]):
            if got==0 or np.all(np.sum((pos[:got]-c)**2,1)>DMIN*DMIN):
                pos[got]=c; got+=1
                if got>=A: break
        tries+=1
    pos[:,:2]-=pos[:,:2].mean(0); return pos
def mc_event(s,b):
    R,a,A=WS[s]; NA=sample_nuc(R,a,A); NB=sample_nuc(R,a,A); NA[:,0]-=b/2; NB[:,0]+=b/2; d2=SIGNN/np.pi
    hit=((NA[:,0][:,None]-NB[:,0][None,:])**2+(NA[:,1][:,None]-NB[:,1][None,:])**2)<=d2
    pA=hit.any(1); pB=hit.any(0); Np=int(pA.sum()+pB.sum())
    if Np<2: return None
    px=np.concatenate([NA[pA,0],NB[pB,0]]); py=np.concatenate([NA[pA,1],NB[pB,1]]); px-=px.mean(); py-=py.mean()
    sx2=np.mean(px*px); sy2=np.mean(py*py); ax_,ay_=np.sqrt(sx2),np.sqrt(sy2)
    phis=np.linspace(0,2*np.pi,12,endpoint=False); Ls=[]
    for i in np.random.choice(len(px),size=min(60,len(px)),replace=False):
        x0,y0=px[i],py[i]
        for ph in phis:
            cx,cy=np.cos(ph),np.sin(ph); Aq=(cx/(2*ax_))**2+(cy/(2*ay_))**2
            Bq=2*(x0*cx/(2*ax_)**2+y0*cy/(2*ay_)**2); Cq=(x0/(2*ax_))**2+(y0/(2*ay_))**2-1; disc=Bq*Bq-4*Aq*Cq
            if disc>0 and Aq>0:
                tt=(-Bq+np.sqrt(disc))/(2*Aq)
                if tt>0: Ls.append(tt)
    return Np,np.mean(Ls) if Ls else 0.0
def mc_geom(s,nevt=500):
    R,a,A=WS[s]; bmax=(2*R+5*a); rows=[]
    for _ in range(nevt):
        e=mc_event(s,bmax*np.sqrt(np.random.uniform()))
        if e: rows.append(e)
    arr=np.array(rows); return arr[:,0].mean(),arr[:,1].mean()
print("MC-Glauber geometry (incl. Ar, Kr)...")
np.random.seed(0); MC={s:mc_geom(s,450) for s in WS}
Npart={s:MC[s][0] for s in WS}; Lm={s:MC[s][1] for s in WS}
G={s:(Npart[s]/Npart["PbPb"])**(1/3) for s in WS}
for s in WS: print(f"  {s:5s} Npart={Npart[s]:6.1f}  Npart^1/3 ratio={G[s]:.3f}")

# ---------------- data ----------------
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

def fit_subset(systems,nsteps=5000):
    def logp(th):
        k,n,b=th
        if not(0<k<12 and 0<=n<=4.5 and -0.5<=b<=1.5): return -np.inf
        ll=-0.5*((n-2)/1.5)**2
        for s in systems:
            pt,raa,_=DAT[s];dpt=k*fs[s]*G[s]**n*np.power(pt,b);r=raa-(pt/(pt+dpt))**a_local(pt,SQRTS[s]);ll+=-0.5*r@cho_solve(CH[s],r)
        return ll
    rng=np.random.default_rng(0); p0=np.array([2.,1.8,.3])+0.02*rng.standard_normal((40,3))
    sm=emcee.EnsembleSampler(40,3,logp); sm.run_mcmc(p0,nsteps,progress=False)
    return sm.get_chain(discard=2000,thin=10,flat=True)

# ---- (1) LOSO physics ----
print("\n== Leave-one-system-out PHYSICS test ==")
full=fit_subset(ORDER); nf=np.percentile(full[:,1],[16,50,84])
print(f"  full 4-system: n={nf[1]:.2f} (+{nf[2]-nf[1]:.2f}/-{nf[1]-nf[0]:.2f})")
loso={}
for drop in ORDER:
    sub=[s for s in ORDER if s!=drop]; ch=fit_subset(sub); q=np.percentile(ch[:,1],[16,50,84])
    loso[drop]={"n":[float(x) for x in q]}; print(f"  drop {drop:5s}: n={q[1]:.2f} (+{q[2]-q[1]:.2f}/-{q[1]-q[0]:.2f})  [3 systems]")
ns=[loso[d]["n"][1] for d in ORDER]
print(f"  -> n spread across LOSO: [{min(ns):.2f},{max(ns):.2f}]  (full={nf[1]:.2f})")

# ---- (2) predictions for Ar+Ar, Kr+Kr ----
print("\n== Predictions for future systems (sqrt(s)=5.36 TeV assumed) ==")
def predict(sys,pg,sqrts=5.36):
    fsp=(sqrts/5.02)**0.31; curves=[]
    for k,n,b in full[::4]:
        dpt=k*fsp*G[sys]**n*np.power(pg,b); curves.append((pg/(pg+dpt))**a_local(pg,sqrts))
    c=np.array(curves); return np.percentile(c,[16,50,84],0)
pg=np.logspace(np.log10(8),np.log10(100),50)
PRED={}
for sys in ("ArAr","KrKr"):
    lo,med,hi=predict(sys,pg)
    at10=predict(sys,np.array([10.0]))[:,0]; at30=predict(sys,np.array([30.0]))[:,0]
    PRED[sys]={"Npart":float(Npart[sys]),"G":float(G[sys]),
               "RAA_10":[float(at10[1]),float(at10[1]-at10[0]),float(at10[2]-at10[1])],
               "RAA_30":[float(at30[1]),float(at30[1]-at30[0]),float(at30[2]-at30[1])]}
    print(f"  {sys}: <Npart>={Npart[sys]:.0f}  R_AA(10GeV)={at10[1]:.2f}+{at10[2]-at10[1]:.2f}-{at10[1]-at10[0]:.2f}  R_AA(30)={at30[1]:.2f}")

# ---- (3) posterior-predictive p-value ----
print("\n== Posterior-predictive check ==")
rng=np.random.default_rng(1); chi2_obs=[]; chi2_rep=[]
for k,n,b in full[rng.integers(0,len(full),400)]:
    co=0.; cr=0.
    for s in ORDER:
        pt,raa,cov=DAT[s]; mu=(pt/(pt+k*fs[s]*G[s]**n*np.power(pt,b)))**a_local(pt,SQRTS[s])
        r=raa-mu; co+=r@cho_solve(CH[s],r)
        rep=rng.multivariate_normal(mu,cov); rr=rep-mu; cr+=rr@cho_solve(CH[s],rr)
    chi2_obs.append(co); chi2_rep.append(cr)
ppp=float(np.mean(np.array(chi2_rep)>=np.array(chi2_obs)))
print(f"  posterior-predictive p-value = {ppp:.2f}  (0.5 ideal; >0.05 acceptable)")

json.dump(dict(mc_geometry={s:{"Npart":float(Npart[s]),"G":float(G[s])} for s in WS},
               full_n=[float(x) for x in nf], loso=loso,
               loso_spread=[float(min(ns)),float(max(ns))],
               predictions=PRED, ppc_pvalue=ppp),
          open("/tmp/out/loso_predictions.json","w"),indent=1)

# ---- figure ----
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
# left: LOSO stability
ys=np.arange(len(ORDER)+1); labels=["full (4 sys)"]+[f"drop {d}" for d in ORDER]
vals=[nf[1]]+[loso[d]["n"][1] for d in ORDER]
los=[nf[1]-nf[0]]+[loso[d]["n"][1]-loso[d]["n"][0] for d in ORDER]
his=[nf[2]-nf[1]]+[loso[d]["n"][2]-loso[d]["n"][1] for d in ORDER]
ax[0].errorbar(vals,ys,xerr=[los,his],fmt="o",color="#1f4e79",ms=7,capsize=4)
ax[0].axvspan(2-0.0,2+0.0,color="green",alpha=0); ax[0].axvline(2,ls="--",c="green",label="radiative $n=2$")
ax[0].axvline(nf[1],ls=":",c="#1f4e79",alpha=.6,label=f"full fit {nf[1]:.2f}")
ax[0].set_yticks(ys); ax[0].set_yticklabels(labels); ax[0].set_xlabel("extracted exponent $n$")
ax[0].set_title("Leave-one-system-out stability"); ax[0].set_xlim(1.0,2.6); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3,axis="x")
# right: predictions
for sys,c in (("ArAr","#2e7d32"),("KrKr","#b7791f")):
    lo,med,hi=predict(sys,pg); ax[1].fill_between(pg,lo,hi,color=c,alpha=.25)
    ax[1].plot(pg,med,color=c,lw=2,label=f"{sys[:2]}+{sys[2:]} (A={WS[sys][2]}, pred.)")
for s,c in (("OO","#555"),("PbPb","#000")):
    pt,raa,cov=DAT[s]; ax[1].errorbar(pt,raa,yerr=np.sqrt(np.diag(cov)),fmt="o",ms=3,color=c,alpha=.6,label=f"{s} (data)")
ax[1].axhline(1,ls="--",lw=.7,c="grey"); ax[1].set_xscale("log"); ax[1].set_xlabel("$p_T$ [GeV]"); ax[1].set_ylabel("$R_{AA}$")
ax[1].set_ylim(0,1.15); ax[1].set_title("Predictions for future systems (calibrated band)"); ax[1].legend(fontsize=8,loc="lower right"); ax[1].grid(alpha=.25,which="both")
fig.suptitle(f"Physics robustness & predictions  (PPC $p$-value = {ppp:.2f})",fontsize=12,y=1.01)
fig.tight_layout(); fig.savefig("/tmp/out/PHYS_loso_predictions.png",dpi=150,bbox_inches="tight")
print("\nsaved PHYS_loso_predictions.png + loso_predictions.json")
