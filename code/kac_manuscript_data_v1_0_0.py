#!/usr/bin/env python3
"""
kac_manuscript_data_v1_0_0.py -- DISCLOSED DEVELOPMENT WORK. Zero offsets.

Closes the provenance gap for numbers that are already asserted in the
manuscript but exist only as printed output.  Section 5 of
hyman2026obstruction_v1_2_0.tex states f* ~ 0.318, C/Cbar = 1.05,
D/Dbar = 1.66, a sixth moment 3.58 times Maxwellian and 57% of the
two-sigma mass retained; none of those had a saved generator.  Section 6's
first experiment, which supplies the paper's best figure, was likewise
stdout-only.

PART A  finite-N invariance sweep -> manuscript_ks_sweep.csv, Figure 2.
        KS distance of v^2/N from its exact uniform-on-sphere marginal
        Beta(1/2,(N-1)/2), for three rules.  Random pairing is the null
        control; a fixed state-independent matching isolates state
        dependence from determinism.  No projection: the rotation conserves
        energy exactly, so the chain sits on the sphere.

PART B  the Section 5 counterexample -> manuscript_hierarchy.csv.
        The blend rule sends a fraction f of the particles to sorted pairing
        among themselves and the rest to extremal pairing among themselves.
        Locate f* where C/Cbar = 1, then measure D/Dbar and the stationary
        moments there.

Precision on output follows the house policy: 3 significant figures for
means, biases, MSEs and ratios; 2 for variances.  Raw CSVs keep full
precision, and rounding happens only at presentation.
"""

import csv
import numpy as np
from scipy import stats

V_TH, DT, N_STEPS = 1.0, 0.05, 800
BASE = 813000000 + 950000
TAIL_GAUSS = 0.045500263896358


# ------------------------------------------------------------------ PART A
def step_sphere(v, rule, rng):
    n_chains, n_part = v.shape
    if rule == "sorted":
        o = np.argsort(np.abs(v), axis=1, kind="stable")
    elif rule == "random":
        o = np.argsort(rng.random((n_chains, n_part)), axis=1)
    elif rule == "fixed":
        o = np.tile(np.arange(n_part), (n_chains, 1))
    else:
        raise ValueError(rule)
    ia, ib = o[:, 0::2], o[:, 1::2]
    va = np.take_along_axis(v, ia, axis=1)
    vb = np.take_along_axis(v, ib, axis=1)
    r = np.hypot(va, vb)
    t = 2.0 * np.pi * rng.random(r.shape)
    np.put_along_axis(v, ia, r * np.cos(t), axis=1)
    np.put_along_axis(v, ib, r * np.sin(t), axis=1)
    return v


def ks_sweep(n_part, rule, seed, chains=400, burn=3000, collect=60, gap=25):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(chains, n_part))
    v *= np.sqrt(n_part / np.sum(v * v, axis=1, keepdims=True))
    for _ in range(burn):
        step_sphere(v, rule, rng)
    pool = []
    for _ in range(collect):
        for _ in range(gap):
            step_sphere(v, rule, rng)
        pool.append((v * v / n_part).ravel())
    s = np.concatenate(pool)
    return float(stats.kstest(s, "beta", args=(0.5, (n_part - 1) / 2.0)).statistic)


# ------------------------------------------------------------------ PART B
def blend_pairs(v, frac, rng):
    n = v.size
    o = np.argsort(np.abs(v), kind="stable")
    n_sim = int(round(frac * n)) // 2 * 2
    pick = rng.permutation(n)
    sim = o[np.sort(pick[:n_sim])]
    ext = o[np.sort(pick[n_sim:])]
    a, b = [sim[0::2]], [sim[1::2]]
    h = ext.size // 2
    if h:
        a.append(ext[:h]); b.append(ext[::-1][:h])
    return np.concatenate(a), np.concatenate(b)


def ratios(v, frac, rng):
    n = v.size
    w = v * np.sqrt(n / np.dot(v, v))
    ia, ib = blend_pairs(w, frac, rng)
    a2, b2 = w[ia] ** 2, w[ib] ** 2
    Q, Q6 = float(np.sum(w ** 4)), float(np.sum(w ** 6))
    C = float(np.sum(a2 * b2))
    D = float(np.sum(a2 * b2 * (a2 + b2)))
    return C / ((n * n - Q) / (2 * (n - 1))), D / ((n * Q - Q6) / (n - 1))


def mean_ratios(n_part, frac, draws=16, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        v = rng.normal(size=n_part)
        v -= v.mean(); v *= np.sqrt(n_part / np.dot(v, v))
        out.append(ratios(v, frac, rng))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean()


def stationary(n_part, frac, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n_part)
    v -= v.mean(); v *= np.sqrt(n_part / np.dot(v, v))
    for _ in range(N_STEPS):
        ia, ib = blend_pairs(v, frac, rng)
        t = 2.0 * np.pi * rng.random(ia.size)
        r = np.hypot(v[ia], v[ib])
        v[ia] = r * np.cos(t); v[ib] = r * np.sin(t)
        v -= v.mean(); v *= np.sqrt(n_part / np.dot(v, v))
    return (float(np.mean(v ** 4)) / 3.0,
            float(np.mean(v ** 6)) / 15.0,
            float(np.mean(np.abs(v) > 2.0)) / TAIL_GAUSS)


def main():
    print("=" * 74)
    print("PART A  finite-N invariance sweep")
    print("=" * 74)
    print(f"{'N':>5} {'KS random':>12} {'KS sorted':>12} {'KS fixed':>12}")
    rows = []
    for n_part in (2, 4, 6, 8, 16, 32, 64):
        kr = ks_sweep(n_part, "random", BASE + 100 + n_part)
        ks = ks_sweep(n_part, "sorted", BASE + 200 + n_part)
        kf = ks_sweep(n_part, "fixed", BASE + 300 + n_part)
        print(f"{n_part:>5} {kr:>12.5f} {ks:>12.5f} {kf:>12.5f}")
        rows.append((n_part, kr, ks, kf))
    with open("manuscript_ks_sweep.csv", "w", newline="\n") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "ks_random", "ks_sorted", "ks_fixed"])
        w.writerows(rows)

    print("\n" + "=" * 74)
    print("PART B  the Section 5 counterexample")
    print("=" * 74)
    n_part = 2048
    grid = [0.28, 0.30, 0.32, 0.34, 0.36]
    print(f"{'f':>7} {'C/Cbar':>9} {'D/Dbar':>9}")
    pts = []
    for f in grid:
        c, d = mean_ratios(n_part, f, seed=BASE + int(f * 1000))
        print(f"{f:>7.3f} {c:>9.4f} {d:>9.4f}")
        pts.append((f, c, d))
    below = [p for p in pts if p[1] <= 1][-1]
    above = [p for p in pts if p[1] > 1][0]
    f_star = below[0] + (above[0] - below[0]) * (1 - below[1]) / (above[1] - below[1])
    c_s, d_s = mean_ratios(n_part, f_star, draws=32, seed=BASE + 7)
    m4, m6, tail = np.mean([stationary(n_part, f_star, BASE + 40 + 10 * k)
                            for k in range(6)], axis=0)
    print(f"\n  f* = {f_star:.4f}   C/Cbar = {c_s:.3f}   D/Dbar = {d_s:.3f}")
    print(f"  stationary: m4 = {m4:.3f}   m6 = {m6:.3f}   tail/Gaussian = {tail:.3f}")
    with open("manuscript_hierarchy.csv", "w", newline="\n") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerows([("f_star", f_star), ("c_over_cbar", c_s), ("d_over_dbar", d_s),
                     ("m4_over_maxwellian", m4), ("m6_over_maxwellian", m6),
                     ("tail_over_gaussian", tail)])
    print("\n  manuscript claims: f* ~ 0.318, C/Cbar = 1.05, D/Dbar = 1.66,")
    print("  m6 = 3.58, tail retained 57%")


if __name__ == "__main__":
    main()
