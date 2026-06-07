import os,glob,json,warnings; warnings.filterwarnings("ignore")
import numpy as np, yaml, emcee
from scipy.linalg import cho_factor, cho_solve
NEW="/tmp/real4/HEPData-ins3123773-v1-yaml"; XE=glob.glob("/tmp/real/HEPData-ins1692558*/")[0]
SQRTS={"OO":5.36,"NeNe":5.36,"XeXe":5.44,"PbPb":5.02}; ORDER=["OO","NeNe","XeXe","PbPb"]
APOLY=np.load("/tmp/a_poly.npy"); a_local=lambda pt,s:(lambda l:APOLY[0]*l*l+APOLY[1]*l+APOLY[2])(np.log(np.asarray(pt)*5.02/s))
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
def fit_n(G):
    def logp(th):
        k,n,b=th
        if not(0<k<12 and 0<=n<=4.5 and -0.5<=b<=1.5): return -np.inf
        ll=-0.5*((n-2)/1.5)**2
        for s in ORDER:
            pt,raa,_=DAT[s];dpt=k*fs[s]*G[s]**n*np.power(pt,b);r=raa-(pt/(pt+dpt))**a_local(pt,SQRTS[s]);ll+=-0.5*r@cho_solve(CH[s],r)
        return ll
    p0=np.array([2.,1.8,.3])+0.02*np.random.default_rng(0).standard_normal((40,3))
    sm=emcee.EnsembleSampler(40,3,logp); sm.run_mcmc(p0,4000,progress=False)
    return np.percentile(sm.get_chain(discard=1500,thin=10,flat=True)[:,1],50)
Gold=json.load(open("/tmp/out/mc_glauber.json"))["mc_ratios"]["Npart13"]
Ghi=json.load(open("/tmp/out/mc_glauber_hires.json"))["Npart13_ratio"]
n_old=fit_n(Gold); n_hi=fit_n(Ghi)
print(f"n (450-600 evt geometry) = {n_old:.3f}")
print(f"n (5000 evt geometry)    = {n_hi:.3f}")
print(f"shift = {abs(n_hi-n_old):.3f}")
