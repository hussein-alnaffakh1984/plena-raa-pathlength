import json, numpy as np, time
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
    ax_,ay_=np.sqrt(np.mean(px*px)),np.sqrt(np.mean(py*py)); phis=np.linspace(0,2*np.pi,12,endpoint=False); Ls=[]
    for i in np.random.choice(len(px),size=min(60,len(px)),replace=False):
        x0,y0=px[i],py[i]
        for ph in phis:
            cx,cy=np.cos(ph),np.sin(ph); Aq=(cx/(2*ax_))**2+(cy/(2*ay_))**2
            Bq=2*(x0*cx/(2*ax_)**2+y0*cy/(2*ay_)**2); Cq=(x0/(2*ax_))**2+(y0/(2*ay_))**2-1; disc=Bq*Bq-4*Aq*Cq
            if disc>0 and Aq>0:
                tt=(-Bq+np.sqrt(disc))/(2*Aq)
                if tt>0: Ls.append(tt)
    return Np,np.mean(Ls) if Ls else 0.0
def mc_geom(s,nevt):
    R,a,A=WS[s]; bmax=(2*R+5*a); rows=[]
    for _ in range(nevt):
        e=mc_event(s,bmax*np.sqrt(np.random.uniform()))
        if e: rows.append(e)
    arr=np.array(rows); return arr[:,0].mean(),arr[:,0].std()/np.sqrt(len(arr)),arr[:,1].mean()
np.random.seed(1); t0=time.time(); NEV=5000
res={}
for s in WS:
    Np,Nperr,L=mc_geom(s,NEV); res[s]={"Npart":Np,"Npart_err":Nperr,"L":L}
    print(f"{s:5s} Npart={Np:7.2f}+-{Nperr:.2f}  L={L:.3f}  ({time.time()-t0:.0f}s)")
ratios={s:(res[s]["Npart"]/res["PbPb"]["Npart"])**(1/3) for s in WS}
json.dump({"nevt":NEV,"abs":res,"Npart13_ratio":ratios},open("/tmp/out/mc_glauber_hires.json","w"),indent=1)
# compare to old 450-600 event ratios
old=json.load(open("/tmp/out/mc_glauber.json"))["mc_ratios"]["Npart13"]
print("\n=== ratio change vs low-stats (Npart^1/3) ===")
maxd=0
for s in ["OO","NeNe","XeXe","PbPb"]:
    d=abs(ratios[s]-old[s]); maxd=max(maxd,d); print(f"  {s:5s}: hires={ratios[s]:.3f} old={old[s]:.3f} diff={d:.4f}")
print(f"  MAX change = {maxd:.4f} ({100*maxd:.2f}%)")
