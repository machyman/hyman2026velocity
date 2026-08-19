#!/usr/bin/env python3
"""
kac_tradeoff_v1_0_0.py -- DISCLOSED DEVELOPMENT WORK. Zero campaign offsets.

CORRECTION to the experiment as first proposed.  I proposed sweeping the
locality dose and measuring "the variance reduction factor".  That is
ill-posed: the reduction contains no RQMC, so there is no variance reduction
in it to measure.  The measurable and more informative quantity is the
BIAS-VARIANCE decomposition of ordinary estimators as the pairing becomes
more velocity-correlated:

  bias      = how far the stationary value of an observable sits from its
              Maxwellian value  (the equilibrium damage the theorem predicts)
  spread    = replicate-to-replicate variability of that observable
              (what a practitioner sees as "converged")

The question that matters for practice is not "how much variance is saved"
but WHICH OBSERVABLES CARRY THE DAMAGE.  Density, momentum and energy are
pinned by construction (the projection, and the rotation's exact pair-energy
conservation), so a scheme can be badly wrong while every routinely-monitored
diagnostic looks perfect.  This script measures a moment ladder to find out
where the damage actually lives.

Observables at t = 40, each normalized so the Maxwellian value is 1.0:
    m2   = <v^2> / v_th^2                     (expected pinned)
    m4   = <v^4> / (3 v_th^4)
    m6   = <v^6> / (15 v_th^6)
    tail = P(|v| > 2 v_th) / 0.0455
plus C/C-bar for the rule, evaluated on the initial Maxwellian, tying each
row back to the hypothesis of Theorem 4'.
"""

import time

import numpy as np

from kac_reduction_v1_0_0 import V_TH, N_STEPS, RED_SEED_BASE, stratified_unit_sample
from kac_reduction_v1_1_0 import make_pairs

TAIL_GAUSS = 0.045500263896358


def c_ratio(v, pairing, rng):
    """C / C-bar on the given state: the hypothesis of Theorem 4'."""
    n = v.size
    w = v * np.sqrt(n / np.dot(v, v))          # normalize sum w^2 = N
    ia, ib = make_pairs(w, pairing, rng)
    C = float(np.sum(w[ia] ** 2 * w[ib] ** 2))
    Q = float(np.sum(w ** 4))
    return C / ((n * n - Q) / (2.0 * (n - 1)))


def run_rep(n_part, pairing, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, V_TH, size=n_part)
    v -= v.mean()
    v *= np.sqrt(n_part * V_TH * V_TH / np.dot(v, v))
    cr = c_ratio(v, pairing, np.random.default_rng(seed + 13))
    for _ in range(N_STEPS):
        ia, ib = make_pairs(v, pairing, rng)
        theta = 2.0 * np.pi * stratified_unit_sample(ia.size, 1, rng)
        r = np.hypot(v[ia], v[ib])
        v[ia] = r * np.cos(theta)
        v[ib] = r * np.sin(theta)
        v -= v.mean()
        v *= np.sqrt(n_part * V_TH * V_TH / np.dot(v, v))
    return dict(
        c_ratio=cr,
        m2=float(np.mean(v ** 2)) / V_TH ** 2,
        m4=float(np.mean(v ** 4)) / (3 * V_TH ** 4),
        m6=float(np.mean(v ** 6)) / (15 * V_TH ** 6),
        tail=float(np.mean(np.abs(v) > 2 * V_TH)) / TAIL_GAUSS,
    )


def main():
    n_part, n_reps = 2048, 8
    widths = [2, 8, 32, 128, 512, n_part]
    t0 = time.time()
    print("=== BIAS-VARIANCE ACROSS THE LOCALITY DOSE (Np=2048, 8 reps, t=40) ===")
    print("Every observable normalized so the Maxwellian value is 1.000.")
    print("'spread' is the replicate coefficient of variation, in percent:")
    print("what a practitioner reads as convergence.\n")
    print(f"{'w':>6} {'C/C-bar':>8} | {'m2':>16} {'m4':>16} {'m6':>16} {'tail':>16}")
    rows = []
    for wi, w in enumerate(widths):
        pairing = f"block:{w}"
        res = [run_rep(n_part, pairing, RED_SEED_BASE + 100000 + 1000 * wi + 10 * r)
               for r in range(n_reps)]
        cr = np.mean([r["c_ratio"] for r in res])
        cells = []
        for k in ("m2", "m4", "m6", "tail"):
            a = np.array([r[k] for r in res])
            cells.append(f"{a.mean():7.3f}+/-{100*a.std(ddof=1)/max(abs(a.mean()),1e-12):5.1f}%")
        print(f"{w:>6} {cr:8.3f} | " + " ".join(f"{c:>16}" for c in cells))
        rows.append((w, cr, [np.array([r[k] for r in res]) for k in ("m2", "m4", "m6", "tail")]))
    print(f"\n  [wall clock {time.time()-t0:.1f}s]")

    print("\n=== READ ===")
    w2 = rows[0]; wr = rows[-1]
    print(f"  strict adjacency (w=2, C/C-bar={w2[1]:.2f}) vs random (w=Np, C/C-bar={wr[1]:.2f}):")
    for i, k in enumerate(("m2", "m4", "m6", "tail")):
        b_s = w2[2][i].mean() - 1.0
        b_r = wr[2][i].mean() - 1.0
        cv_s = 100 * w2[2][i].std(ddof=1) / max(abs(w2[2][i].mean()), 1e-12)
        cv_r = 100 * wr[2][i].std(ddof=1) / max(abs(wr[2][i].mean()), 1e-12)
        print(f"    {k:>4}: bias {b_s:+8.3f} (sorted) vs {b_r:+8.3f} (random);"
              f"  spread {cv_s:6.2f}% vs {cv_r:6.2f}%")


if __name__ == "__main__":
    main()
