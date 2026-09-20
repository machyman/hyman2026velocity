#!/usr/bin/env python3
"""Test the two-particle stationary law of the sorted rule across N.

Section 9 of hyman2026velocity conjectures that the sorted rule drives the
Kac chain to a state in which two particles carry the whole energy on a
circle of radius sqrt(N) with uniform phase and every other particle is at
rest.  That law is an assertion about the stationary measure and it makes
exact predictions for the even moments:

    m_2k  =  (N/2)^(k-1) (2k-1)!! / k!

so that m4 = 3N/4 and m6 = 5N^2/8, against a true equilibrium on the sphere
sum v_i^2 = N whose moments are N^k (2k-1)!! / (N(N+2)...(N+2k-2)), which
tend to the Maxwellian values 3 and 15.  The predicted moments therefore grow
without bound in N while the correct ones do not.

This script measures m4, m6, the number of moving particles and max|v|/sqrt(N)
for the sorted rule over a range of N, under two seeds, and reports each
against the closed form.  It also checks the coordinate identification
s = 1 - E1/N, which reconciles the E1 reduction found at N = 4 with the
earlier prediction of a one-dimensional chain on s in [1/2, 1]: E1 -> 0 is
s -> 1, the upper endpoint.

The law is identified numerically.  No proof of the energy decay is offered
here, and Section 9 states the conjecture as a conjecture.

Input   none (self-contained simulation)
Output  n4_law_sweep.csv

Determinism.  Seeded from SEEDS below; reproducible on any machine with the
same numpy.  No figure is drawn, so there is no timestamp to pin.

Usage:  python3 kac_n4_law_sweep_v1_0_0.py [--outdir DIR] [--quick]
"""
import argparse
import csv
import os
from math import factorial

import numpy as np

N_VALUES = (4, 16, 64, 256, 1024, 2048)
SEEDS = (813950011, 813950012)
SAMPLE_BUDGET = 30000          # chains chosen so chains*N is about constant
BURN, COLLECT, GAP = 2000, 20, 20
REST = 1e-6                    # |v| below REST*sqrt(N) counts as at rest


def dfact(n):
    """Double factorial (2k-1)!!."""
    r = 1
    for i in range(n, 0, -2):
        r *= i
    return r


def sphere_moment(N, k):
    """Exact 2k-th moment of the uniform measure on sum v_i^2 = N."""
    num, den = N ** k * dfact(2 * k - 1), 1
    for j in range(k):
        den *= (N + 2 * j)
    return num / den


def law_moment(N, k):
    """The conjectured two-particle law: (N/2)^(k-1) (2k-1)!! / k!."""
    return (N / 2.0) ** (k - 1) * dfact(2 * k - 1) / factorial(k)


def step(v, rng):
    """One sorted-rule Kac step.  Pairs by ascending |v|; the rotation
    conserves each pair's energy, so the chain stays on the sphere."""
    o = np.argsort(np.abs(v), axis=1, kind="stable")
    ia, ib = o[:, 0::2], o[:, 1::2]
    va = np.take_along_axis(v, ia, axis=1)
    vb = np.take_along_axis(v, ib, axis=1)
    r = np.hypot(va, vb)
    t = 2.0 * np.pi * rng.random(r.shape)
    np.put_along_axis(v, ia, r * np.cos(t), axis=1)
    np.put_along_axis(v, ib, r * np.sin(t), axis=1)
    return v


def run(N, seed, burn, collect, gap):
    """Return (m4, m6, mean moving count, max|v|/sqrt(N), mean E1/N)."""
    chains = max(4, SAMPLE_BUDGET // N)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(chains, N))
    v *= np.sqrt(N / np.sum(v * v, axis=1, keepdims=True))
    for _ in range(burn):
        step(v, rng)
    pool = []
    for _ in range(collect):
        for _ in range(gap):
            step(v, rng)
        pool.append(v.copy())
    s = np.concatenate([p.ravel() for p in pool])
    s2 = s * s
    moving = float(np.mean([(np.abs(p) > REST * np.sqrt(N)).sum(axis=1).mean()
                            for p in pool]))
    # E1: energy held by every particle except the two fastest.  The law says
    # this vanishes, which in the s coordinate is s = 1 - E1/N -> 1.
    e1 = float(np.mean([np.sort(p * p, axis=1)[:, :-2].sum(axis=1).mean()
                        for p in pool]))
    return (float(np.mean(s2 * s2)), float(np.mean(s2 * s2 * s2)),
            moving, float(np.max(np.abs(s))) / np.sqrt(N), e1 / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    burn, collect, gap = (200, 6, 10) if a.quick else (BURN, COLLECT, GAP)

    rows = []
    hdr = (f"{'N':>6} {'m4':>10} {'3N/4':>10} {'dev%':>7} {'m6':>13} "
           f"{'5N^2/8':>13} {'dev%':>7} {'moving':>7} {'max/sqrtN':>10} {'s':>8}")
    print(hdr)
    for N in N_VALUES:
        acc = [run(N, sd, burn, collect, gap) for sd in SEEDS]
        m4 = sum(x[0] for x in acc) / len(acc)
        m6 = sum(x[1] for x in acc) / len(acc)
        mv = sum(x[2] for x in acc) / len(acc)
        mx = max(x[3] for x in acc)
        sc = 1.0 - sum(x[4] for x in acc) / len(acc)
        p4, p6 = law_moment(N, 2), law_moment(N, 3)
        d4, d6 = 100 * (m4 - p4) / p4, 100 * (m6 - p6) / p6
        spread4 = 100 * abs(acc[0][0] - acc[1][0]) / m4
        spread6 = 100 * abs(acc[0][1] - acc[1][1]) / m6
        print(f"{N:>6} {m4:>10.6g} {p4:>10.6g} {d4:>+7.2f} {m6:>13.6g} "
              f"{p6:>13.6g} {d6:>+7.2f} {mv:>7.2f} {mx:>10.4f} {sc:>8.5f}")
        rows.append([N, f"{m4:.10g}", f"{p4:.10g}", f"{d4:.4f}", f"{spread4:.4f}",
                     f"{m6:.10g}", f"{p6:.10g}", f"{d6:.4f}", f"{spread6:.4f}",
                     f"{mv:.4f}", f"{mx:.6f}", f"{sc:.8f}",
                     f"{sphere_moment(N,2):.10g}", f"{sphere_moment(N,3):.10g}"])

    with open(os.path.join(a.outdir, "n4_law_sweep.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "m4_measured", "m4_law", "m4_dev_pct", "m4_seed_spread_pct",
                    "m6_measured", "m6_law", "m6_dev_pct", "m6_seed_spread_pct",
                    "moving_particles", "max_abs_v_over_sqrtN", "s_coordinate",
                    "m4_true_equilibrium", "m6_true_equilibrium"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
