#!/usr/bin/env python3
"""
(A) REAL Monte-Carlo Glauber (pure Python, no ROOT) — independent cross-check
    of the optical-Glauber geometry used in the analysis.
Samples nucleons from Woods-Saxon, applies sigma_nn collision criterion,
computes <Npart>, <Ncoll>, transverse area S, eccentricity, and exit path
length <L>, then compares minimum-bias ratios with the optical Glauber.
"""
import os, json, numpy as np
np.random.seed(12345)
OUT = "/tmp/out"
# Woods-Saxon (R fm, a fm), mass number A, max-centrality fraction used
WS = {"OO": (2.608, 0.513, 16), "NeNe": (2.80, 0.55, 20), "XeXe": (5.40, 0.59, 129), "PbPb": (6.62, 0.546, 208)}
CMAX = {"OO": 1.0, "NeNe": 1.0, "XeXe": 0.80, "PbPb": 1.0}
SIGNN = 7.0            # inelastic nucleon-nucleon cross section [fm^2] (70 mb @ 5 TeV)
DMIN = 0.4            # hard-core exclusion [fm]

def sample_nucleus(R, a, A, rmax=None):
    rmax = rmax or (R + 8*a)
    pos = np.empty((A, 3)); got = 0; tries = 0
    while got < A and tries < 200000:
        nbatch = 4*(A-got)
        r = rmax*np.cbrt(np.random.uniform(size=nbatch))
        keep = np.random.uniform(size=nbatch) < 1.0/(1.0+np.exp((r-R)/a))
        r = r[keep]
        if len(r) == 0: tries += 1; continue
        ct = np.random.uniform(-1, 1, len(r)); ph = np.random.uniform(0, 2*np.pi, len(r)); st = np.sqrt(1-ct*ct)
        cand = np.column_stack([r*st*np.cos(ph), r*st*np.sin(ph), r*ct])
        for c in cand:
            if got == 0 or np.all(np.sum((pos[:got]-c)**2, axis=1) > DMIN*DMIN):
                pos[got] = c; got += 1
                if got >= A: break
        tries += 1
    pos[:, :2] -= pos[:, :2].mean(0)      # recenter transverse
    return pos

def one_event(sysA, sysB, b):
    RA, aA, A = WS[sysA]; RB, aB, B = WS[sysB]
    NA = sample_nucleus(RA, aA, A); NB = sample_nucleus(RB, aB, B)
    NA[:, 0] -= b/2; NB[:, 0] += b/2
    d2 = SIGNN/np.pi
    dx = NA[:, 0][:, None]-NB[:, 0][None, :]; dy = NA[:, 1][:, None]-NB[:, 1][None, :]
    hit = (dx*dx+dy*dy) <= d2
    Ncoll = int(hit.sum()); pA = hit.any(1); pB = hit.any(0); Npart = int(pA.sum()+pB.sum())
    if Npart < 2: return None
    px = np.concatenate([NA[pA, 0], NB[pB, 0]]); py = np.concatenate([NA[pA, 1], NB[pB, 1]])
    px -= px.mean(); py -= py.mean()
    sx2 = np.mean(px*px); sy2 = np.mean(py*py); sxy = np.mean(px*py)
    S = np.pi*np.sqrt(max(sx2*sy2-sxy*sx2*0, 1e-9)); S = 4*np.pi*np.sqrt(max(sx2*sy2, 1e-9))
    ecc = np.sqrt((sy2-sx2)**2+4*sxy*sxy)/(sx2+sy2+1e-9)
    # exit path length: average over participants & directions of distance to leave the participant zone
    # density proxy: 2D KDE-free -> use participant points; L = mean over sources of straight-line length
    # within an ellipse of rms (sqrt(sx2),sqrt(sy2)) (Glauber-standard transverse extent)
    ax_, ay_ = np.sqrt(sx2), np.sqrt(sy2); phis = np.linspace(0, 2*np.pi, 12, endpoint=False)
    Ls = []
    idx = np.random.choice(len(px), size=min(60, len(px)), replace=False)
    for i in idx:
        x0, y0 = px[i], py[i]
        for ph in phis:
            cx, cy = np.cos(ph), np.sin(ph)
            # exit when ((x0+t cx)/(2ax))^2+((y0+t cy)/(2ay))^2 = 1  (2-sigma ellipse ~ medium edge)
            Ax = (cx/(2*ax_))**2+(cy/(2*ay_))**2; Bx = 2*(x0*cx/(2*ax_)**2+y0*cy/(2*ay_)**2)
            Cx = (x0/(2*ax_))**2+(y0/(2*ay_))**2-1
            disc = Bx*Bx-4*Ax*Cx
            if disc > 0 and Ax > 0:
                t = (-Bx+np.sqrt(disc))/(2*Ax)
                if t > 0: Ls.append(t)
    L = np.mean(Ls) if Ls else 0.0
    return Npart, Ncoll, S, ecc, L

def mb_average(sys, nevt=600):
    RA, aA, A = WS[sys]; bmax = (2*RA+5*aA)*np.sqrt(CMAX[sys])
    rows = []
    for _ in range(nevt):
        b = bmax*np.sqrt(np.random.uniform())     # dN/db ~ b
        e = one_event(sys, sys, b)
        if e: rows.append(e)
    arr = np.array(rows)  # Npart,Ncoll,S,ecc,L
    # minimum-bias = average over events (already b-weighted by sampling)
    return dict(Npart=arr[:, 0].mean(), Ncoll=arr[:, 1].mean(), S=arr[:, 2].mean(),
                ecc=arr[:, 3].mean(), L=arr[:, 4].mean(), nevt=len(arr),
                Npart_e=arr[:, 0].std()/np.sqrt(len(arr)), L_e=arr[:, 4].std()/np.sqrt(len(arr)))

print("REAL MC-Glauber (pure Python). sigma_nn=%.1f fm^2, d_min=%.1f fm\n" % (SIGNN, DMIN))
mc = {}
for s in ["OO", "NeNe", "XeXe", "PbPb"]:
    mc[s] = mb_average(s, nevt=500)
    m = mc[s]; print(f"  {s:5s} <Npart>={m['Npart']:6.1f}±{m['Npart_e']:.1f}  <Ncoll>={m['Ncoll']:6.1f}  "
                      f"S={m['S']:5.1f}  ecc={m['ecc']:.3f}  <L>={m['L']:.2f}±{m['L_e']:.2f}  (n={m['nevt']})")

# ratios vs PbPb
mcGN = {s: (mc[s]["Npart"]/mc["PbPb"]["Npart"])**(1/3) for s in mc}
mcGL = {s: mc[s]["L"]/mc["PbPb"]["L"] for s in mc}
# optical Glauber (from results.json)
opt = json.load(open(os.path.join(OUT, "results.json")))["geometry"]
optGN = {s: (opt["Npart"][s]/opt["Npart"]["PbPb"])**(1/3) for s in mc}
optGL = {s: opt["L"][s]/opt["L"]["PbPb"] for s in mc}
print("\n  geometry RATIOS vs PbPb  (MC-Glauber vs optical):")
print(f"  {'sys':5s} {'Npart^1/3 MC':>12s} {'opt':>6s} {'|':>2s} {'<L> MC':>8s} {'opt':>6s}")
for s in ["OO", "NeNe", "XeXe", "PbPb"]:
    print(f"  {s:5s} {mcGN[s]:12.3f} {optGN[s]:6.3f}  |  {mcGL[s]:8.3f} {optGL[s]:6.3f}")
maxdGN = max(abs(mcGN[s]-optGN[s]) for s in mc); maxdGL = max(abs(mcGL[s]-optGL[s]) for s in mc)
print(f"\n  max |MC-optical| ratio diff:  Npart^1/3={maxdGN:.3f}   <L>={maxdGL:.3f}")
json.dump({"mc": {s: {k: float(mc[s][k]) for k in ("Npart", "Ncoll", "S", "ecc", "L")} for s in mc},
           "mc_ratios": {"Npart13": mcGN, "L": mcGL}, "opt_ratios": {"Npart13": optGN, "L": optGL},
           "max_diff_Npart13": maxdGN, "max_diff_L": maxdGL},
          open(os.path.join(OUT, "mc_glauber.json"), "w"), indent=1, default=float)
print("\nsaved mc_glauber.json")
