#!/usr/bin/env python3
"""
kac_theory_v1_0_0.py
====================
DISCLOSED DEVELOPMENT WORK. Zero campaign offsets (seed base 813000000).

Question this probe exists to answer: is the invariance failure a FINITE-N
statement, or a large-N emergent one?

Why it matters.  Everything so far is numerical evidence at Np = 2048 and
8192.  If state-dependent pairing breaks the invariant measure at every
N >= 4, the claim is theorem-shaped and the paper can carry a proof.  If it
only appears at large N, the claim is an asymptotic/mean-field statement and
must be written as one.

Setup (deliberately cleaner than the reduction).  Pure Kac: the rotation
    (v_a, v_b) -> (R cos t, R sin t),  R^2 = v_a^2 + v_b^2
conserves sum(v^2) EXACTLY, so with no projection at all the chain lives
exactly on the sphere S^{N-1}(sqrt(N)).  The Kac equilibrium is the uniform
measure on that sphere, whose single-component marginal is exact:
    v_i^2 / N  ~  Beta(1/2, (N-1)/2).
No mean-removal is applied here -- that would be an extra projection off the
sphere and would confound the test.

The test.  Initialize uniformly on the sphere, run, and measure the KS
distance between the empirical law of v_i^2/N and that exact Beta.  Under
uniform random matching the uniform measure is invariant (the rotation is
chosen independently of the state), so the KS distance must stay at the
sampling floor -- this is the null control, carrying the same correlation
structure as the treatment.  Under sorted pairing, any drift is an
invariance failure at that N.

Sharp prediction from the tiling argument.  A piecewise rotation -- rotation
R_i applied on region A_i -- preserves uniform measure iff the images
{R_i(A_i)} again tile the sphere a.e.  For N = 2 there is only one perfect
matching, so the rule cannot depend on the state and invariance must hold.
For N >= 4 the sorted rule applies different rotations on different
orderings of |v| and the images are not expected to tile.  So: NO failure at
N = 2, failure at every N >= 4, with N = 4 the minimal counterexample.
"""

import argparse
import time

import numpy as np
from scipy import stats


def step_vec(v, pairing, rng):
    """One Kac step for all chains at once.  v has shape (n_chains, n_part).
    No projection: the rotation conserves each row's energy exactly."""
    n_chains, n_part = v.shape
    if pairing == "sorted":
        order = np.argsort(np.abs(v), axis=1, kind="stable")
    elif pairing == "fixed":
        # A FIXED, state-independent matching: always pair (0,1), (2,3), ...
        # A fixed rotation preserves the uniform measure on the sphere, so
        # invariance must hold here even though the rule is deterministic
        # and highly structured.  This isolates STATE DEPENDENCE as the
        # culprit, as against structure or determinism.
        # (Caveat: this chain is not ergodic -- each pair's energy is
        # conserved forever -- but the test is of invariance, not mixing,
        # and it is initialized from the uniform measure.)
        order = np.tile(np.arange(n_part), (n_chains, 1))
    elif pairing == "random":
        order = np.argsort(rng.random((n_chains, n_part)), axis=1)
    else:
        raise ValueError(pairing)

    idx_a, idx_b = order[:, 0::2], order[:, 1::2]
    va = np.take_along_axis(v, idx_a, axis=1)
    vb = np.take_along_axis(v, idx_b, axis=1)
    radius = np.hypot(va, vb)
    theta = 2.0 * np.pi * rng.random(radius.shape)
    np.put_along_axis(v, idx_a, radius * np.cos(theta), axis=1)
    np.put_along_axis(v, idx_b, radius * np.sin(theta), axis=1)
    return v


def ks_to_beta(samples, n_part):
    """KS distance between the empirical law of v^2/N and its exact
    uniform-on-sphere marginal Beta(1/2, (N-1)/2)."""
    return float(stats.kstest(samples, "beta", args=(0.5, (n_part - 1) / 2.0)).statistic)


def run_case(n_part, pairing, n_chains, n_burn, n_collect, gap, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_chains, n_part))
    v *= np.sqrt(n_part / np.sum(v * v, axis=1, keepdims=True))   # exact uniform on the sphere

    for _ in range(n_burn):
        step_vec(v, pairing, rng)

    pool = []
    for _ in range(n_collect):
        for _ in range(gap):
            step_vec(v, pairing, rng)
        pool.append((v * v / n_part).ravel())
    return ks_to_beta(np.concatenate(pool), n_part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=400)
    ap.add_argument("--burn", type=int, default=3000)
    ap.add_argument("--collect", type=int, default=60)
    ap.add_argument("--gap", type=int, default=25)
    args = ap.parse_args()

    print("=== T1 FINITE-N INVARIANCE TEST ===")
    print("KS distance of v^2/N from the exact uniform-on-sphere Beta marginal.")
    print("'random' is the null control: uniform is exactly invariant there.")
    print(f"{'N':>5} {'KS random (null)':>18} {'KS sorted':>12} {'ratio':>9}   verdict")
    t0 = time.time()
    for n_part in (2, 4, 6, 8, 16, 32, 64):
        ks_r = run_case(n_part, "random", args.chains, args.burn, args.collect, args.gap,
                        813000000 + 50000 + n_part)
        ks_s = run_case(n_part, "sorted", args.chains, args.burn, args.collect, args.gap,
                        813000000 + 60000 + n_part)
        ratio = ks_s / ks_r if ks_r > 0 else np.inf
        verdict = "INVARIANT" if ratio < 3 else "FAILS"
        print(f"{n_part:>5} {ks_r:>18.5f} {ks_s:>12.5f} {ratio:>9.1f}   {verdict}")
    print(f"  [wall clock {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
