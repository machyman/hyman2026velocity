#!/usr/bin/env python3
"""
kac_hierarchy_v1_0_0.py -- DISCLOSED DEVELOPMENT WORK. Zero campaign offsets.

The fourth-moment condition C = C-bar was shown to be NECESSARY BUT NOT
SUFFICIENT: blend:0.3177 passes it and is badly non-Maxwellian.  So the
diagnostic I recommended is one-sided.  This script asks whether a SECOND
member of the hierarchy catches the counterexample.

Derivation (same three lines as Lemma 2, one moment up).  With
Q6(v) = sum_i v_i^6 and E[cos^6] = E[sin^6] = 5/16, so E[cos^6+sin^6] = 5/8:

    R^6 = (v_a^2 + v_b^2)^3 = v_a^6 + v_b^6 + 3 v_a^2 v_b^2 (v_a^2 + v_b^2)

    E_theta[Q6(Tv) | v] = (5/8) ( Q6(v) + 3 D(P,v) ),
        D(P,v) = sum_P v_a^2 v_b^2 (v_a^2 + v_b^2).                    (L2')

So the sixth moment depends on the pairing through a DIFFERENT functional D,
not through C.  The exchangeable average, by the same argument as Lemma 3
(sum_i v_i^2 = N):

    D-bar = (1/(2(N-1))) sum_{i != j} (v_i^4 v_j^2 + v_i^2 v_j^4)
          = (1/(N-1)) sum_{i != j} v_i^4 v_j^2
          = (N * Q4 - Q6) / (N - 1).                                   (L3')

If D/D-bar departs from 1 for the compliant counterexample, the hierarchy's
second member catches what the first misses, and the diagnostic becomes
two-sided in a usable way: check C/C-bar AND D/D-bar.
"""

import numpy as np

from kac_reduction_v1_0_0 import V_TH, RED_SEED_BASE
from kac_reduction_v1_2_0 import make_pairs_e


def functionals(v, pairing, rng):
    """Return (C/C-bar, D/D-bar) on the given state, normalized sum v^2 = N."""
    n = v.size
    w = v * np.sqrt(n / np.dot(v, v))
    ia, ib = make_pairs_e(w, pairing, rng)
    wa2, wb2 = w[ia] ** 2, w[ib] ** 2
    C = float(np.sum(wa2 * wb2))
    D = float(np.sum(wa2 * wb2 * (wa2 + wb2)))
    Q4 = float(np.sum(w ** 4))
    Q6 = float(np.sum(w ** 6))
    C_bar = (n * n - Q4) / (2.0 * (n - 1))
    D_bar = (n * Q4 - Q6) / (n - 1)
    return C / C_bar, D / D_bar


def mean_functionals(n_part, pairing, n_draws=16, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        v = rng.normal(0.0, V_TH, size=n_part)
        v -= v.mean()
        v *= np.sqrt(n_part * V_TH ** 2 / np.dot(v, v))
        out.append(functionals(v, pairing, rng))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean()


def main():
    n_part = 2048
    rng = np.random.default_rng(RED_SEED_BASE + 250000)

    print("=== CHECK 1: L3' closed form for D-bar against random-pairing Monte Carlo ===")
    v = rng.normal(0.0, V_TH, size=n_part)
    v *= np.sqrt(n_part / np.dot(v, v))
    Q4 = float(np.sum(v ** 4)); Q6 = float(np.sum(v ** 6))
    D_bar = (n_part * Q4 - Q6) / (n_part - 1)
    samp = []
    for _ in range(200):
        p = rng.permutation(n_part)
        a2, b2 = v[p[0::2]] ** 2, v[p[1::2]] ** 2
        samp.append(float(np.sum(a2 * b2 * (a2 + b2))))
    print(f"  D-bar closed form {D_bar:.6f}   Monte Carlo {np.mean(samp):.6f}   "
          f"rel {abs(D_bar-np.mean(samp))/D_bar:.2e}")

    print("\n=== CHECK 2: uniform-measure fixed point of the sixth-moment map ===")
    Q4_u = 3.0 * n_part ** 2 / (n_part + 2)
    Q6_u = 15.0 * n_part ** 3 / ((n_part + 2) * (n_part + 4))
    D_bar_u = (n_part * Q4_u - Q6_u) / (n_part - 1)
    lhs = 0.625 * (Q6_u + 3 * D_bar_u)
    print(f"  E_sigma[Q6] = 15N^3/((N+2)(N+4)) = {Q6_u:.6f}")
    print(f"  (5/8)(Q6 + 3 D-bar)              = {lhs:.6f}   rel {abs(lhs-Q6_u)/Q6_u:.2e}")
    print("  (a fixed point confirms L2' and L3' are mutually consistent)")

    print("\n=== CHECK 3: does D/D-bar catch the compliant counterexample? ===")
    print(f"{'rule':>26} {'C/C-bar':>9} {'D/D-bar':>9}   verdict")
    for label, pairing in (
        ("block:2 (sorted)", "block:2"),
        ("blend:0.0 (extremal)", "blend:0.0"),
        ("blend:0.3177 (C-compliant)", "blend:0.3177"),
        ("block:1024 (compliant+clean)", "block:1024"),
        ("block:2048 (random)", "block:2048"),
    ):
        c, d = mean_functionals(n_part, pairing, seed=17)
        okc, okd = abs(c - 1) < 0.10, abs(d - 1) < 0.10
        verdict = ("passes both" if okc and okd else
                   "CAUGHT by D only" if okd is False and okc else
                   "caught by C")
        print(f"{label:>26} {c:9.3f} {d:9.3f}   {verdict}")


if __name__ == "__main__":
    main()
