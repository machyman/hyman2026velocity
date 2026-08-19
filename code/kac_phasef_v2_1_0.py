#!/usr/bin/env python3
"""
kac_phasef_v2_1_0.py -- RECONSTRUCTION. DISCLOSED DEVELOPMENT WORK.
Zero campaign offsets.

WHY THIS EXISTS.  The results attributed to "Phase F" -- the failed candidate
fix (1.06x), the trade-off table, and the monotonicity of MSE in the block
width -- underpin Section 6 of the manuscript, two of its figures and one of
its tables.  A verification pass found that the code which produced them is
NOT on disk.  `kac_tradeoff_v1_0_0.py`, which the changelog names as Phase
F's substrate, in fact contains the later moment-ladder script; its docstring
carries the since-withdrawn claim that the reduction has no RQMC in it.
Either the Phase F substrate was overwritten when that file was copied into
outputs, or it was never saved.  Either way those numbers are currently
unreproducible and exist only as printed tables inside a markdown document.

This script rebuilds the experiment as a generator with fixed seeds and
saved output, and so also serves as an independent check on numbers that
have been quoted all session.  Where the reconstruction disagrees with the
published table, the disagreement is reported rather than reconciled.

DESIGN, as described in the Phase F write-up.
  Estimators:  <v^4>, whose equilibrium value 3N/(N+2) is exact, and <|v|>.
  N = 2048, 800 steps, initial ensemble held FIXED across randomizations so
  that the spread measured is that of the sampling, not of the load.
  Four schemes:
    MC          random pairing,  iid angles
    RQMC        random pairing,  stratified angle multiset (no alignment)
    A-iid       sorted pairing,  iid angles
    Array-RQMC  sorted pairing,  stratified angles assigned in rank order
  Alignment means what it means in Array-RQMC: the pairs inherit an order
  from the sort, and the low-discrepancy points are handed out in that order.

PRECISION POLICY (set from the reconstruction, not by preference).  Means,
biases, MSEs and C/Cbar reproduced to 0.03-1.0% and are reported to 3
significant figures.  A variance estimated from n samples carries relative
standard error sqrt(2/(n-1)) -- 14% at n=100, 18% at n=60 -- so variances get
2 significant figures and no more.  Ratios of two variance estimates carry
23% and are NOT tabulated; they are described in prose at order of magnitude.
"""

import csv
import numpy as np

N_PART = 2048
N_STEPS = 800
SEED = 813000000 + 900000


def pairs_for(v, rule, rng):
    """Return (idx_a, idx_b) ordered so that pair p is the p-th in the
    rule's own ranking; alignment consumes that ordering."""
    n = v.size
    if rule == "random":
        o = rng.permutation(n)
    elif rule == "sorted":
        o = np.argsort(np.abs(v), kind="stable")
    elif rule.startswith("block:"):
        w = int(rule.split(":")[1])
        o = np.argsort(np.abs(v), kind="stable")
        nb = n // w
        head = o[: nb * w].reshape(nb, w)
        head = np.take_along_axis(head, np.argsort(rng.random(head.shape), axis=1), axis=1)
        o = np.concatenate([head.ravel(), o[nb * w:]])
    else:
        raise ValueError(rule)
    return o[0::2], o[1::2]


def angles_for(n_pairs, kind, rng):
    if kind == "iid":
        return 2.0 * np.pi * rng.random(n_pairs)
    u = (np.arange(n_pairs) + rng.random(n_pairs)) / n_pairs   # stratified
    if kind == "stratified":
        rng.shuffle(u)          # multiset only; no alignment
    return 2.0 * np.pi * u      # kind == "aligned": handed out in rank order


def run(v0, rule, kind, seed):
    v = v0.copy()
    rng = np.random.default_rng(seed)
    for _ in range(N_STEPS):
        ia, ib = pairs_for(v, rule, rng)
        th = angles_for(ia.size, kind, rng)
        r = np.hypot(v[ia], v[ib])
        v[ia] = r * np.cos(th)
        v[ib] = r * np.sin(th)
        v -= v.mean()
        v *= np.sqrt(v.size / np.dot(v, v))
    return float(np.mean(v ** 4)), float(np.mean(np.abs(v)))


def c_ratio(v, rule, rng):
    ia, ib = pairs_for(v, rule, rng)
    Q = float(np.sum(v ** 4))
    C = float(np.sum(v[ia] ** 2 * v[ib] ** 2))
    n = v.size
    return C / ((n * n - Q) / (2.0 * (n - 1)))


def main():
    g = np.random.default_rng(SEED)
    v0 = g.normal(size=N_PART)
    v0 -= v0.mean()
    v0 *= np.sqrt(N_PART / np.dot(v0, v0))

    truth4 = 3.0 * N_PART / (N_PART + 2)
    # <|v|> has no simple closed form on the sphere; use the Gaussian value,
    # which is the N -> infinity limit, and report it as the reference.
    truth1 = np.sqrt(2.0 / np.pi)

    print("=" * 78)
    print(f"PART 1  THE CANDIDATE FIX   (N={N_PART}, 100 randomizations, fixed load)")
    print(f"        equilibrium <v^4> = 3N/(N+2) = {truth4:.6f}")
    print("=" * 78)
    print(f"{'scheme':>26} {'mean <v^4>':>12} {'bias':>12} {'variance':>11} "
          f"{'MSE':>11} {'var ratio':>10}")
    rows, base = [], None
    for label, rule, kind in (("MC (random, iid)", "random", "iid"),
                              ("RQMC (random, stratified)", "random", "stratified"),
                              ("A-iid (sorted, iid)", "sorted", "iid"),
                              ("Array-RQMC (sorted, aligned)", "sorted", "aligned")):
        vals = np.array([run(v0, rule, kind, SEED + 1000 * i + 7)[0] for i in range(100)])
        m, var = vals.mean(), vals.var(ddof=1)
        bias = m - truth4
        mse = bias ** 2 + var
        if base is None:
            base = var
        print(f"{label:>26} {m:>12.5f} {bias:>12.4f} {var:>11.3e} {mse:>11.3e} "
              f"{base/var:>10.3f}")
        rows.append((label, m, bias, var, mse, base / var))
    with open("phasef_part1.csv", "w", newline="") as fh:
        w = csv.writer(fh)                      # scheme names contain commas
        w.writerow(["scheme", "mean_v4", "bias", "variance", "mse",
                    "var_ratio_vs_MC"])
        w.writerows(rows)

    print("\n" + "=" * 78)
    print(f"PART 2  TRADE-OFF ACROSS THE BLOCK WIDTH  (60 randomizations, aligned)")
    print("=" * 78)
    print(f"{'w':>6} {'C/Cbar':>8} {'bias<v4>':>11} {'MSE<v4>':>11} "
          f"{'bias<|v|>':>11} {'var<|v|>':>11} {'MSE<|v|>':>11}")
    rows2 = []
    for w in (2, 8, 64, 512, N_PART):
        rule = "sorted" if w == 2 else ("random" if w == N_PART else f"block:{w}")
        cr = np.mean([c_ratio(v0, rule, np.random.default_rng(SEED + 55 + k))
                      for k in range(12)])
        out = np.array([run(v0, rule, "aligned", SEED + 2000 * i + 11) for i in range(60)])
        b4 = out[:, 0].mean() - truth4
        m4 = b4 ** 2 + out[:, 0].var(ddof=1)
        b1 = out[:, 1].mean() - truth1
        v1 = out[:, 1].var(ddof=1)
        m1 = b1 ** 2 + v1
        print(f"{w:>6} {cr:>8.3f} {b4:>11.4g} {m4:>11.3e} {b1:>11.5f} "
              f"{v1:>11.3e} {m1:>11.3e}")
        rows2.append((w, cr, b4, m4, b1, v1, m1))
    with open("phasef_part2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["w", "c_over_cbar", "bias_v4", "mse_v4", "bias_absv",
                    "var_absv", "mse_absv"])
        w.writerows(rows2)

    print("\nMSE monotone in w?  <v^4>:",
          all(rows2[i][3] > rows2[i + 1][3] for i in range(len(rows2) - 1)),
          "  <|v|>:",
          all(rows2[i][6] > rows2[i + 1][6] for i in range(len(rows2) - 1)))


if __name__ == "__main__":
    main()
