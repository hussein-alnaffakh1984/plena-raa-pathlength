#!/usr/bin/env python3
"""External cross-experiment validation (reviewer #12): overlay ALICE Xe-Xe
R_AA (Phys.Lett.B 788, ins1672790) on the XeXe posterior-predictive band of the
joint CMS fit, WITHOUT using ALICE in the fit. ALICE provides per-centrality
R_AA (0-5%..70-80%); we form an effective 0-80% (minimum-bias-like) R_AA by
N_coll-weighting, with <N_coll> per class from our from-scratch MC Glauber."""
import os,glob,json,warnings; warnings.filterwarnings("ignore")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ALICE="/tmp/alice/HEPData-ins1672790-v3-yaml/Fig4_RXeXe_5440GeV.yaml"
APOLY=np.load("/tmp/a_poly.npy"); a_local=lambda pt,s:(lambda l:APOLY[0]*l*l+APOLY[1]*l+APOLY[2])(np.log(np.asarray(pt)*5.02/s))
G=json.load(open("/tmp/out/mc_glauber.json"))["mc_ratios"]["Npart13"]["XeXe"]
ch=np.load("/tmp/out/chain_Npart_effective.npy")   # (N,>=3): kappa,n,beta
fs=(5.44/5.02)**0.31

# ---------- ALICE R_AA per centrality ----------
d=yaml.safe_load(open(ALICE)); iv=d["independent_variables"][0]["values"]; dv=d["dependent_variables"]
pt=np.array([0.5*(x["low"]+x["high"]) if "low" in x else x["value"] for x in iv])
def num(x):
    try: return float(x)
    except: return np.nan
CLASSES=[]
for v in dv:
    cen=[q["value"] for q in v["qualifiers"] if q["name"]=="CENT"][0]
    raa=np.array([num(val["value"]) for val in v["values"]],dtype=float)
    err=np.array([np.sqrt(sum((float(e["symerror"]) if "symerror" in e else
          0.5*(abs(float(e["asymerror"]["plus"]))+abs(float(e["asymerror"]["minus"]))))**2
          for e in val.get("errors",[]))) if num(val["value"])==num(val["value"]) else np.nan
          for val in v["values"]])
    lo,hi=[float(x) for x in cen.replace("pct","").split("-")]
    CLASSES.append({"cen":cen,"lo":lo,"hi":hi,"width":hi-lo,"raa":raa,"err":err})

# ---------- MC Glauber Xe-Xe: <N_coll> per centrality class ----------
print("MC Glauber Xe-Xe for <N_coll> per centrality ...")
R,a,A=5.40,0.59,129; SIGNN=7.0; DMIN=0.4
def sample_nuc():
    rmax=R+8*a; pos=np.empty((A,3)); got=0; tries=0
    while got<A and tries<200000:
        r=rmax*np.cbrt(np.random.uniform(size=4*(A-got))); r=r[np.random.uniform(size=len(r))<1/(1+np.exp((r-R)/a))]
        if len(r)==0: tries+=1; continue
        ct=np.random.uniform(-1,1,len(r)); ph=np.random.uniform(0,2*np.pi,len(r)); st=np.sqrt(1-ct*ct)
        for c in np.column_stack([r*st*np.cos(ph),r*st*np.sin(ph),r*ct]):
            if got==0 or np.all(np.sum((pos[:got]-c)**2,1)>DMIN*DMIN): pos[got]=c; got+=1
            if got>=A: break
        tries+=1
    return pos
def event(b):
    NA=sample_nuc(); NB=sample_nuc(); NA[:,0]-=b/2; NB[:,0]+=b/2; d2=SIGNN/np.pi
    dx=NA[:,0][:,None]-NB[:,0][None,:]; dy=NA[:,1][:,None]-NB[:,1][None,:]; hit=(dx*dx+dy*dy)<=d2
    Ncoll=int(hit.sum()); Npart=int(hit.any(1).sum()+hit.any(0).sum())
    return Npart,Ncoll
np.random.seed(7); NEV=4000; bmax=2*R+5*a
ev=[event(bmax*np.sqrt(np.random.uniform())) for _ in range(NEV)]
ev=np.array([e for e in ev if e[0]>=2]); 
order=np.argsort(-ev[:,0])  # sort by Npart descending (centrality proxy)
ev_sorted=ev[order]; npart_s=ev_sorted[:,0]; ncoll_s=ev_sorted[:,1]; N=len(ev_sorted)
def mean_ncoll(lo,hi):
    i0,i1=int(lo/100*N),int(hi/100*N); return ncoll_s[i0:i1].mean()
for c in CLASSES: c["Ncoll"]=mean_ncoll(c["lo"],c["hi"])
print("  <N_coll> per class:", {c["cen"]:round(c["Ncoll"],0) for c in CLASSES})

# ---------- effective 0-80% R_AA = sum w_c R_AA,c , w_c = width_c * Ncoll_c ----------
W=np.array([c["width"]*c["Ncoll"] for c in CLASSES]); W/=W.sum()
RAA=np.array([c["raa"] for c in CLASSES]); ERR=np.array([c["err"] for c in CLASSES])
valid=np.isfinite(RAA).all(axis=0)   # keep pT points present in all classes
raa_eff=np.sum(W[:,None]*np.nan_to_num(RAA),axis=0)
err_eff=np.sqrt(np.sum((W[:,None]*np.nan_to_num(ERR))**2,axis=0))
pt_v=pt[valid]; raa_eff=raa_eff[valid]; err_eff=err_eff[valid]
print("  effective 0-80% R_AA at ~10,~20,~30 GeV:",
      [round(raa_eff[np.argmin(abs(pt_v-x))],3) for x in (10,20,30)])

# ---------- XeXe posterior-predictive band (canonical refit, data-driven a) ----------
import emcee, glob as _glob, re as _re
from scipy.linalg import cho_factor, cho_solve
SQRTS={"OO":5.36,"NeNe":5.36,"XeXe":5.44,"PbPb":5.02}; ORDER=["OO","NeNe","XeXe","PbPb"]
NEW="/tmp/real4/HEPData-ins3123773-v1-yaml"; XEd=_glob.glob("/tmp/real/HEPData-ins1692558*/")[0]
def _parse(dd):
    iv=dd["independent_variables"][0]["values"]; dv=dd["dependent_variables"][0]["values"]; rows=[]
    for i,val in enumerate(dv):
        v=iv[i]; ptv=0.5*(float(v["low"])+float(v["high"])) if "low" in v else float(v["value"]); row={"pt":ptv,"raa":float(val["value"])}
        for e in val.get("errors",[]):
            lab=(e.get("label") or "e").lower(); row[lab]=abs(float(e["symerror"])) if "symerror" in e else 0.5*(abs(float(e["asymerror"]["plus"]))+abs(float(e["asymerror"]["minus"])))
        rows.append(row)
    return rows
def _loadn(fn): return _parse(yaml.safe_load(open(f"{NEW}/{fn}")))
def _loadxe():
    Fy={os.path.basename(p):open(p).read() for p in _glob.glob(XEd+"/*.yaml")}; best=None;span=-1
    for dd in yaml.safe_load_all(Fy["submission.yaml"]):
        if isinstance(dd,dict) and "data_file" in dd:
            kw={k["name"]:k.get("values",[]) for k in dd.get("keywords",[])}
            if any("RAA" in str(x).upper() for x in kw.get("observables",[])):
                z=yaml.safe_load(Fy[dd["data_file"]]);cen="n/a"
                for q in z["dependent_variables"][0].get("qualifiers",[]):
                    if "CENT" in str(q.get("name","")).upper(): cen=str(q.get("value",""))
                mm=_re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",cen.lower());sp=float(mm[0][1])-float(mm[0][0]) if mm else 100
                if sp>span: span=sp;best=z
    return _parse(best)
RAW={"OO":_loadn("oo_raa_(coarse_pt_binning).yaml"),"NeNe":_loadn("nene_raa_(coarse_pt_binning).yaml"),
     "PbPb":_loadn("pbpb_raa_(coarse_pt_binning).yaml"),"XeXe":_loadxe()}
Gall=json.load(open("/tmp/out/mc_glauber.json"))["mc_ratios"]["Npart13"]
def _cls(n):
    n=n.lower(); return "u" if "stat" in n else ("f" if any(k in n for k in("taa","lumi","norm","global")) else "p")
def _build(nm,xi=4.0):
    rows=RAW[nm];pts=np.array([r["pt"] for r in rows]);raa=np.array([r["raa"] for r in rows])
    comps=set().union(*[set(r) for r in rows])-{"pt","raa"};nn=len(rows);Cc=np.zeros((nn,nn));D=np.abs(np.subtract.outer(np.arange(nn),np.arange(nn)))
    for c in comps:
        v=np.array([r.get(c,0.) for r in rows]);kk=_cls(c);Cc+=np.diag(v**2) if kk=="u" else (np.outer(v,v) if kk=="f" else np.outer(v,v)*np.exp(-D/xi))
    mm=pts>=8.0;idx=np.where(mm)[0];return pts[mm],raa[mm],Cc[np.ix_(idx,idx)]
DAT={s:_build(s) for s in ORDER}; CHf={s:cho_factor(DAT[s][2]) for s in ORDER}; fsd={s:(SQRTS[s]/5.02)**0.31 for s in ORDER}
def _logp(th):
    k,n,b=th
    if not(0<k<12 and 0<=n<=4.5 and -0.5<=b<=1.5): return -np.inf
    ll=-0.5*((n-2)/1.5)**2
    for s in ORDER:
        pt_,raa_,_=DAT[s];dpt=k*fsd[s]*Gall[s]**n*np.power(pt_,b);r=raa_-(pt_/(pt_+dpt))**a_local(pt_,SQRTS[s]);ll+=-0.5*r@cho_solve(CHf[s],r)
    return ll
p0=np.array([2.,1.8,.3])+0.02*np.random.default_rng(0).standard_normal((40,3))
sm=emcee.EnsembleSampler(40,3,_logp); sm.run_mcmc(p0,4000,progress=False)
chain=sm.get_chain(discard=1500,thin=10,flat=True)
pg=np.logspace(np.log10(8),np.log10(40),30); band=[]
for k,n,b in chain:
    dpt=k*fs*G**n*np.power(pg,b); band.append((pg/(pg+dpt))**a_local(pg,5.44))
band=np.array(band); blo,bmd,bhi=np.percentile(band,[16,50,84],axis=0)
print(f"  refit n_eff (check) = {np.median(chain[:,1]):.3f}; band at 10.8 GeV = {bmd[np.argmin(abs(pg-10.8))]:.3f} (CMS data 0.338)")

# ---------- consistency at pT>=8 (over all valid overlapping points) ----------
m=(pt_v>=8); inside=0; tot=0
for x,y,e in zip(pt_v[m],raa_eff[m],err_eff[m]):
    j=np.argmin(abs(pg-x)); inside+= (blo[j]-e<=y<=bhi[j]+e); tot+=1
ptmax=pt_v[m].max()
print(f"  overlap region: 8 - {ptmax:.0f} GeV; ALICE points consistent with band (within errors): {inside}/{tot}")
json.dump({"Ncoll_per_class":{c["cen"]:float(c["Ncoll"]) for c in CLASSES},
           "raa_eff_080":{f"{x}GeV":float(raa_eff[np.argmin(abs(pt_v-x))]) for x in (8,10,20,30)},
           "consistent_within_errors":f"{inside}/{tot}"},open("/tmp/out/alice_overlay.json","w"),indent=1)

# ---------- figure ----------
fig,ax=plt.subplots(figsize=(7,5))
ax.fill_between(pg,blo,bhi,color="#1f4e79",alpha=.25,label="CMS joint-fit 68% band (Xe+Xe, not using ALICE)")
ax.plot(pg,bmd,color="#1f4e79",lw=1.8)
mm=pt_v<=40
ax.errorbar(pt_v[mm],raa_eff[mm],yerr=err_eff[mm],fmt="o",color="#c0392b",ms=5,capsize=2,
            label="ALICE Xe+Xe 5.44 TeV (eff. 0-80%, $N_{coll}$-weighted)")
ax.axvline(8,ls=":",c="grey",alpha=.7); ax.text(8.2,0.05,"fit region $p_T\\geq8$",fontsize=8,color="grey")
ax.axhline(1,ls="--",lw=.7,c="grey"); ax.set_xscale("log")
ax.set_xlabel("$p_T$ [GeV]"); ax.set_ylabel("$R_{AA}$"); ax.set_ylim(0,1.15); ax.set_xlim(6,42)
ax.set_title("External cross-experiment check: ALICE Xe+Xe vs CMS-fit band")
ax.legend(fontsize=8.5,loc="lower right"); ax.grid(alpha=.25,which="both")
fig.tight_layout(); fig.savefig("/tmp/out/PHYS_alice_overlay.png",dpi=150,bbox_inches="tight")
print("saved PHYS_alice_overlay.png + alice_overlay.json")
