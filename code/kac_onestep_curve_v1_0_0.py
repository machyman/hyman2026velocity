"""E1, curve rule, on the registered PIC load. Declared deviation closed.

NOTE ON WHAT IS BEING TESTED. Lemma 4.2 is a CONDITIONAL identity given v, so
it holds for any state, not only a sigma-draw. It describes the collision step
T alone. The Section 7 campaign applies T and THEN the projection, so what is
tested here is T, which is what the lemma claims.
"""
import numpy as np
from scipy.stats import norm
from hilbertcurve.hilbertcurve import HilbertCurve

ARCHIVE = "discrepancy_campaign_R4_FULL_results_2026-08-12.zip"

def _ensure(npart):
    """Standalone: extract the registered snapshot from its archive if absent."""
    import os, zipfile
    f = f"snapshot_S-C4b_Np{npart}_o74_t0.npz"
    if not os.path.exists(f):
        if not os.path.exists(ARCHIVE):
            raise SystemExit(f"need {f} or the registered archive {ARCHIVE}")
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extract(f)
    return f

def load(npart):
    d = np.load(_ensure(npart))
    x, v = d["x"], d["v"].astype(float)
    return x, v * np.sqrt(npart / np.sum(v * v))

def curve_pairs(x, v, order=10):
    hc = HilbertCurve(order, 2); side = 2**order - 1
    xi = np.clip((x*side).astype(int), 0, side)
    vi = np.clip((norm.cdf(v/v.std())*side).astype(int), 0, side)
    o = np.argsort(np.array(hc.distances_from_points(np.stack([xi, vi], 1))), kind="stable")
    return o[0::2], o[1::2]

for npart in (2048, 8192):
    x, v = load(npart); N = len(v)
    a, b = curve_pairs(x, v)
    assert len(np.unique(np.concatenate([a, b]))) == N, "not a perfect matching"
    va, vb = v[a], v[b]
    R2 = va**2 + vb**2
    Q = float(np.sum(v**4)); C = float(np.sum(va**2 * vb**2))
    Cbar = (N*N - Q)/(2*(N-1))
    pred = 0.75*(Q + 2*C)
    for seed in (20260817, 77770817):
        rng = np.random.default_rng(seed)
        M = 20000
        th = rng.uniform(0, 2*np.pi, size=(M, len(R2)))
        emp = np.sum(R2**2 * (np.cos(th)**4 + np.sin(th)**4), axis=1)
        d = emp.mean() - pred; se = emp.std(ddof=1)/np.sqrt(M)
        print(f"N={npart} seed={seed}  C/Cbar={C/Cbar:.3f}  emp={emp.mean():.4f} "
              f"pred={pred:.4f}  rel={d/pred:+.5%}  d/SE={d/se:+.2f}  "
              f"{'PASS' if abs(d/pred)<0.005 and abs(d/se)<3 else 'FAIL'}")
