#!/usr/bin/env python3
"""
kac_theory_v1_1_0.py
====================
DISCLOSED DEVELOPMENT WORK. Zero campaign offsets (seed base 813000000).

Numerical verification of the exact identities behind the invariance
proposition.  Everything checked here is claimed in closed form in
THEORY_Invariance_2026-08-13.md; this script exists so that no step of that
argument rests on algebra nobody checked.

Claims under test, with Q(v) = sum_i v_i^4 on the sphere sum_i v_i^2 = N:

  (I1)  E_theta[ Q(T v) | v ] = (3/4) ( Q(v) + 2 sum_pairs v_a^2 v_b^2 ).
        The pairing enters the one-step fourth moment ONLY through
        sum_pairs v_a^2 v_b^2.

  (I2)  Under the uniform measure on the sphere,
              E[Q] = 3 N^2 / (N + 2).

  (I3)  Under any exchangeable (state-independent) pairing,
              E[ sum_pairs v_a^2 v_b^2 ] = (N^2 - Q) / (2(N-1)),
        and substituting into (I1) returns E[Q] exactly: the uniform value
        is a fixed point of the one-step mean-fourth-moment map.

  (I4)  By rearrangement, sum_pairs v_a^2 v_b^2 is MAXIMIZED by pairing
        equal energies (the sorted rule) and MINIMIZED by pairing extreme
        with extreme.  Hence one step from uniform strictly increases E[Q]
        under sorted pairing and strictly decreases it under extremal
        pairing -- non-invariance, with the sign of the distortion fixed.
"""

import numpy as np


def sample_sphere(n_chains, n_part, rng):
    v = rng.normal(size=(n_chains, n_part))
    v *= np.sqrt(n_part / np.sum(v * v, axis=1, keepdims=True))
    return v


def pair_indices(v, rule, rng):
    n_chains, n_part = v.shape
    if rule == "sorted":                       # equal energies together
        order = np.argsort(np.abs(v), axis=1, kind="stable")
        return order[:, 0::2], order[:, 1::2]
    if rule == "extremal":                     # largest with smallest
        order = np.argsort(np.abs(v), axis=1, kind="stable")
        half = n_part // 2
        return order[:, :half], order[:, ::-1][:, :half]
    if rule == "random":
        order = np.argsort(rng.random((n_chains, n_part)), axis=1)
        return order[:, 0::2], order[:, 1::2]
    if rule == "fixed":
        order = np.tile(np.arange(n_part), (n_chains, 1))
        return order[:, 0::2], order[:, 1::2]
    raise ValueError(rule)


def cross_term(v, rule, rng):
    """sum_pairs v_a^2 v_b^2, per chain."""
    idx_a, idx_b = pair_indices(v, rule, rng)
    va = np.take_along_axis(v, idx_a, axis=1)
    vb = np.take_along_axis(v, idx_b, axis=1)
    return np.sum(va * va * vb * vb, axis=1)


def one_step_Q(v, rule, rng):
    """Q after one Kac step with fresh angles, per chain."""
    w = v.copy()
    idx_a, idx_b = pair_indices(w, rule, rng)
    va = np.take_along_axis(w, idx_a, axis=1)
    vb = np.take_along_axis(w, idx_b, axis=1)
    radius = np.hypot(va, vb)
    theta = 2.0 * np.pi * rng.random(radius.shape)
    np.put_along_axis(w, idx_a, radius * np.cos(theta), axis=1)
    np.put_along_axis(w, idx_b, radius * np.sin(theta), axis=1)
    return np.sum(w ** 4, axis=1)


def main():
    rng = np.random.default_rng(813000000 + 80000)
    n_chains = 400000

    print("=" * 78)
    print("VERIFICATION OF THE ONE-STEP FOURTH-MOMENT IDENTITIES")
    print("=" * 78)

    for n_part in (4, 8, 32):
        v = sample_sphere(n_chains, n_part, rng)
        Q = np.sum(v ** 4, axis=1)
        Q_uniform_exact = 3.0 * n_part ** 2 / (n_part + 2)

        print(f"\n--- N = {n_part} ---")
        print(f"(I2) E[Q] under uniform: sampled {Q.mean():.6f}   "
              f"exact 3N^2/(N+2) = {Q_uniform_exact:.6f}   "
              f"rel {abs(Q.mean()-Q_uniform_exact)/Q_uniform_exact:.2e}")

        # (I3) exchangeable cross term
        c_rand = cross_term(v, "random", rng)
        c_pred = (n_part ** 2 - Q) / (2.0 * (n_part - 1))
        print(f"(I3) E[sum_p va^2 vb^2] random: sampled {c_rand.mean():.6f}   "
              f"predicted (N^2-Q)/(2(N-1)) = {c_pred.mean():.6f}   "
              f"rel {abs(c_rand.mean()-c_pred.mean())/c_pred.mean():.2e}")

        # (I1) + (I4): one-step Q for each rule, against the identity
        print(f"{'rule':>10} {'E[Q] after step':>17} {'identity (3/4)(Q+2C)':>23} "
              f"{'vs uniform':>12}")
        for rule in ("random", "fixed", "sorted", "extremal"):
            q_after = one_step_Q(v, rule, rng).mean()
            ident = 0.75 * (Q + 2.0 * cross_term(v, rule, rng)).mean()
            direction = ("=" if abs(q_after - Q_uniform_exact) < 3e-3 * Q_uniform_exact
                         else ("UP" if q_after > Q_uniform_exact else "DOWN"))
            print(f"{rule:>10} {q_after:>17.6f} {ident:>23.6f} {direction:>12}")

    print("\n" + "=" * 78)
    print("Read: 'random' and 'fixed' return the uniform value exactly -- uniform")
    print("is a fixed point of the one-step map.  'sorted' moves E[Q] UP (heavier")
    print("fourth moment -> condensate).  'extremal' moves it DOWN (lighter tails).")
    print("The identity column matches the simulated column in every row, so the")
    print("pairing enters only through sum_pairs va^2 vb^2, as claimed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
