#!/usr/bin/env python3
"""
kac_compliance_v1_0_0.py -- DISCLOSED DEVELOPMENT WORK. Zero campaign offsets.

THE QUESTION THIS EXISTS TO SETTLE.

I have been recommending C/C-bar as a practitioner diagnostic: "compute your
pairing's cross term on a Maxwellian load, divide by (N^2-Q)/(2(N-1)), and
if the answer is not 1 your equilibrium is wrong."  Theorem 4' licenses that
direction and only that direction.  The CONVERSE -- if it IS 1, you are fine
-- is untested, and it is the direction a practitioner would actually rely
on.  Shipping the diagnostic without testing sufficiency risks giving false
reassurance, which is worse than shipping no diagnostic at all.

So: is C = C-bar SUFFICIENT for the Maxwellian to be stationary, or merely
NECESSARY?

Two families, both velocity-aware:

  block:w  -- sorted, paired at random inside blocks of w consecutive ranks.
              The damage profile put C/C-bar at 1.714 for w=512 and 0.954
              for w=Np, so compliance in this family lands essentially at
              random pairing.  If so, compliance costs all the locality --
              and locality is exactly what an Array-RQMC sort needs.

  blend:f  -- fraction f of particles energy-MATCHED (adjacent), the rest
              energy-ANTIMATCHED (extremal).  Matched pushes C up,
              antimatched pushes it down, so some f* makes the aggregate
              cross term exactly compliant while BOTH sub-populations remain
              strongly structured.  This is the interesting case: a rule
              that passes the diagnostic without being random.

If the blend rule at f* is Maxwellian, C = C-bar is empirically sufficient
and there is a design principle.  If it is not, the diagnostic is one-sided
and must be published as such.
"""

import time

import numpy as np

from kac_reduction_v1_0_0 import V_TH, N_STEPS, RED_SEED_BASE, stratified_unit_sample
from kac_reduction_v1_2_0 import make_pairs_e

TAIL_GAUSS = 0.045500263896358


def c_ratio(v, pairing, rng):
    n = v.size
    w = v * np.sqrt(n / np.dot(v, v))
    ia, ib = make_pairs_e(w, pairing, rng)
    C = float(np.sum(w[ia] ** 2 * w[ib] ** 2))
    Q = float(np.sum(w ** 4))
    return C / ((n * n - Q) / (2.0 * (n - 1)))


def mean_c_ratio(n_part, pairing, n_draws=12, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        v = rng.normal(0.0, V_TH, size=n_part)
        v -= v.mean()
        v *= np.sqrt(n_part * V_TH ** 2 / np.dot(v, v))
        out.append(c_ratio(v, pairing, rng))
    return float(np.mean(out))


def run_rep(n_part, pairing, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, V_TH, size=n_part)
    v -= v.mean()
    v *= np.sqrt(n_part * V_TH ** 2 / np.dot(v, v))
    for _ in range(N_STEPS):
        ia, ib = make_pairs_e(v, pairing, rng)
        theta = 2.0 * np.pi * stratified_unit_sample(ia.size, 1, rng)
        r = np.hypot(v[ia], v[ib])
        v[ia] = r * np.cos(theta)
        v[ib] = r * np.sin(theta)
        v -= v.mean()
        v *= np.sqrt(n_part * V_TH ** 2 / np.dot(v, v))
    return (float(np.mean(v ** 2)) / V_TH ** 2,
            float(np.mean(v ** 4)) / (3 * V_TH ** 4),
            float(np.mean(v ** 6)) / (15 * V_TH ** 6),
            float(np.mean(np.abs(v) > 2 * V_TH)) / TAIL_GAUSS)


def main():
    n_part, n_reps = 2048, 8
    t0 = time.time()

    print("=== STEP 1: locate C/C-bar = 1 in each family (on the Maxwellian load) ===")
    print(f"{'rule':>14} {'C/C-bar':>9}")
    for w in (512, 1024, 2048):
        print(f"{'block:'+str(w):>14} {mean_c_ratio(n_part, f'block:{w}'):9.3f}")
    grid = [0.20, 0.25, 0.30, 0.32, 0.35, 0.40, 0.50]
    vals = [(f, mean_c_ratio(n_part, f"blend:{f}")) for f in grid]
    for f, c in vals:
        print(f"{'blend:'+str(f):>14} {c:9.3f}")

    below = [(f, c) for f, c in vals if c <= 1.0][-1]
    above = [(f, c) for f, c in vals if c > 1.0][0]
    f_star = below[0] + (above[0] - below[0]) * (1.0 - below[1]) / (above[1] - below[1])
    f_star = round(f_star, 4)
    c_star = mean_c_ratio(n_part, f"blend:{f_star}", n_draws=24, seed=7)
    print(f"\n  interpolated f* = {f_star}  ->  C/C-bar = {c_star:.4f}")

    print("\n=== STEP 2: is the compliant rule Maxwellian? (Np=2048, 8 reps, t=40) ===")
    print("All observables normalized so the Maxwellian value is 1.000.")
    print(f"{'rule':>16} {'C/C-bar':>8} {'m2':>9} {'m4':>10} {'m6':>12} {'tail':>9}")
    for label, pairing in ((f"blend:{f_star} (compliant)", f"blend:{f_star}"),
                           ("blend:0.0 (extremal)", "blend:0.0"),
                           ("block:1024 (compliant)", "block:1024"),
                           ("block:2 (sorted)", "block:2"),
                           ("block:2048 (random)", "block:2048")):
        res = np.array([run_rep(n_part, pairing, RED_SEED_BASE + 150000 + 10 * r)
                        for r in range(n_reps)])
        m = res.mean(axis=0)
        cr = mean_c_ratio(n_part, pairing, n_draws=8, seed=11)
        print(f"{label:>16} {cr:8.3f} {m[0]:9.4f} {m[1]:10.3f} {m[2]:12.3f} {m[3]:9.3f}")

    print(f"\n  [wall clock {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
