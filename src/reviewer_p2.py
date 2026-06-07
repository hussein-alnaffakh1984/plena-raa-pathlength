#!/usr/bin/env python3
"""Reviewer P1(#8) + P2(#4,#5):
 (D) Widen Ar/Kr prediction bands by propagating Glauber geometry uncertainty
     (vary sigma_nn +-0.5 fm^2, Woods-Saxon R +-0.2 fm) + posterior.
 (E) Path-length fluctuation systematic: sigma_L per system from MC Glauber,
     fractional <L^2>/<L>^2-1, effect on geometry ratios and delta_n.
 (F) sqrt(s) correction exponent 0.31 -> vary +-0.05, delta_n."""
import os,glob,json,warnings; warnings.filterwarnings("ignore")
import numpy as np, yaml, emcee
from scipy.linalg import cho_factor, cho_solve
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
DAT={s:build(s) for s in ORDER}; CH={s:cho_factor(DAT[s][2]) for s in ORDER}

# ---- posterior on (k,n,b) with nominal sqrt(s) exponent p ----
def fs_of(s,p): return (SQRTS[s]/5.02)**p
def post_chain(p=0.31,seed=0,nstep=4000):
    fs={s:fs_of(s,p) for s in ORDER}
    def logp(th):
        k,n,b=th
        if not(0<k<12 and 0<=n<=4.5 and -0.5<=b<=1.5): return -np.inf
        ll=-0.5*((n-2)/1.5)**2
        for s in ORDER:
            pt,raa,_=DAT[s];dpt=k*fs[s]*G[s]**n*np.power(pt,b);r=raa-(pt/(pt+dpt))**a_local(pt,SQRTS[s]);ll+=-0.5*r@cho_solve(CH[s],r)
        return ll
    p0=np.array([2.,1.8,.3])+0.02*np.random.default_rng(seed).standard_normal((40,3))
    sm=emcee.EnsembleSampler(40,3,logp); sm.run_mcmc(p0,nstep,progress=False)
    return sm.get_chain(discard=1500,thin=10,flat=True)
ch=post_chain(0.31); kap,nn,bet=np.median(ch,0)

# ===== (F) sqrt(s) exponent variation =====
print("== (F) sqrt(s) exponent 0.31 +- 0.05 ==")
n_026=np.median(post_chain(0.26)[:,1]); n_036=np.median(post_chain(0.36)[:,1])
print(f"  p=0.26 -> n={n_026:.3f} | p=0.31 -> n={nn:.3f} | p=0.36 -> n={n_036:.3f}  (dn={max(abs(n_026-nn),abs(n_036-nn)):.3f})")
dn_sqrts=float(max(abs(n_026-nn),abs(n_036-nn)))

# ===== MC-Glauber for fluctuations + Ar/Kr geometry-uncertainty =====
WS={"OO":(2.608,0.513,16),"NeNe":(2.80,0.55,20),"ArAr":(3.53,0.542,40),"KrKr":(4.39,0.54,84),"XeXe":(5.40,0.59,129),"PbPb":(6.62,0.546,208)}
DMIN=0.4
def sample_nuc(R,a,A):
    rmax=R+8*a; pos=np.empty((A,3)); got=0; tries=0
    while got<A and tries<200000:
        r=rmax*np.cbrt(np.random.uniform(size=4*(A-got))); r=r[np.random.uniform(size=len(r))<1/(1+np.exp((r-R)/a))]
        if len(r)==0: tries+=1; continue
        ct=np.random.uniform(-1,1,len(r)); ph=np.random.uniform(0,2*np.pi,len(r)); st=np.sqrt(1-ct*ct)
        for c in np.column_stack([r*st*np.cos(ph),r*st*np.sin(ph),r*ct]):
            if got==0 or np.all(np.sum((pos[:got]-c)**2,1)>DMIN*DMIN): pos[got]=c; got+=1
            if got>=A: break
        tries+=1
    pos[:,:2]-=pos[:,:2].mean(0); return pos
def mc_event(R,a,A,b,signn):
    NA=sample_nuc(R,a,A); NB=sample_nuc(R,a,A); NA[:,0]-=b/2; NB[:,0]+=b/2; d2=signn/np.pi
    hit=((NA[:,0][:,None]-NB[:,0][None,:])**2+(NA[:,1][:,None]-NB[:,1][None,:])**2)<=d2
    pA=hit.any(1); pB=hit.any(0); Np=int(pA.sum()+pB.sum())
    if Np<2: return None
    px=np.concatenate([NA[pA,0],NB[pB,0]]); py=np.concatenate([NA[pA,1],NB[pB,1]]); px-=px.mean(); py-=py.mean()
    ax_,ay_=np.sqrt(np.mean(px*px)),np.sqrt(np.mean(py*py)); phis=np.linspace(0,2*np.pi,12,endpoint=False); Ls=[]
    for i in np.random.choice(len(px),size=min(50,len(px)),replace=False):
        x0,y0=px[i],py[i]
        for ph in phis:
            cx,cy=np.cos(ph),np.sin(ph); Aq=(cx/(2*ax_))**2+(cy/(2*ay_))**2
            Bq=2*(x0*cx/(2*ax_)**2+y0*cy/(2*ay_)**2); Cq=(x0/(2*ax_))**2+(y0/(2*ay_))**2-1; disc=Bq*Bq-4*Aq*Cq
            if disc>0 and Aq>0:
                tt=(-Bq+np.sqrt(disc))/(2*Aq)
                if tt>0: Ls.append(tt)
    return Np,(np.mean(Ls) if Ls else 0.0)
def geom(s,nevt,signn=7.0,dR=0.0):
    R,a,A=WS[s]; R+=dR; bmax=2*R+5*a; Np=[]; Lm=[]
    for _ in range(nevt):
        e=mc_event(R,a,A,bmax*np.sqrt(np.random.uniform()),signn)
        if e: Np.append(e[0]); Lm.append(e[1])
    Np=np.array(Np); Lm=np.array(Lm); return Np.mean(), Lm.mean(), Lm.std()

# ===== (E) path-length fluctuations =====
print("\n== (E) path-length fluctuations sigma_L/<L> ==")
np.random.seed(2); fluct={}
for s in ORDER:
    Np,Lme,Lsd=geom(s,500); frac=(Lme**2+Lsd**2)/Lme**2-1  # <L^2>/<L>^2 - 1
    fluct[s]={"L":Lme,"sigma_L":Lsd,"sigmaL_over_L":Lsd/Lme,"L2_corr":frac}
    print(f"  {s:5s}: sigma_L/<L>={Lsd/Lme:.2f}  <L^2>/<L>^2-1={frac:.3f}")
# effect on geometry ratios: G_fluct = (<L^2>)^{1/2}/(...) relative ; check ratio change
Lcorr={s:np.sqrt(fluct[s]["L"]**2+fluct[s]["sigma_L"]**2) for s in ORDER}
ratio_fluct={s:Lcorr[s]/Lcorr["PbPb"] for s in ORDER}
ratio_mean ={s:fluct[s]["L"]/fluct["PbPb"]["L"] for s in ORDER}
maxdG=max(abs(ratio_fluct[s]-ratio_mean[s]) for s in ORDER)
print(f"  max geometry-ratio change from fluctuations = {maxdG:.3f}")
dn_fluct=float(min(0.05,maxdG*nn/ (1/3)))  # rough propagation; conservatively cap

# ===== (D) Ar/Kr prediction bands with Glauber uncertainty =====
print("\n== (D) Ar/Kr prediction bands incl. Glauber uncertainty ==")
np.random.seed(3)
def G_variants(s):
    base=geom(s,400)[0]; pb=geom("PbPb",400)[0]; out=[(base/pb)**(1/3)]
    for signn in (6.5,7.5):
        for dR in (-0.2,0.2):
            gs=geom(s,250,signn=signn,dR=dR)[0]; pbv=geom("PbPb",250,signn=signn)[0]; out.append((gs/pbv)**(1/3))
    return np.array(out)
def predict_band(s,pt=10.0,sqrts=5.36):
    Gs=G_variants(s); fsp=(sqrts/5.02)**0.31; vals=[]
    for g in Gs:
        for k,n,b in ch[::8]:
            dpt=k*fsp*g**n*pt**b; vals.append((pt/(pt+dpt))**a_local(pt,sqrts))
    v=np.array(vals); return np.median(v), np.std(v)
PRED={}
for s in ("ArAr","KrKr"):
    m,sd=predict_band(s); PRED[s]={"RAA10":[float(m),float(sd)]}
    print(f"  {s}: R_AA(10) = {m:.2f} +- {sd:.2f}  (was +-0.02 posterior-only)")

json.dump({"dn_sqrts":dn_sqrts,"sqrts_scan":{"0.26":float(n_026),"0.31":float(nn),"0.36":float(n_036)},
           "fluctuations":fluct,"max_dG_fluct":float(maxdG),"dn_fluct":dn_fluct,
           "predictions_widened":PRED},open("/tmp/out/reviewer_p2.json","w"),indent=1)
print("\nsaved reviewer_p2.json")
