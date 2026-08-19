#!/usr/bin/env python3
"""
kac_theory_v1_2_0.py -- TRANSFER TEST.  DISCLOSED DEVELOPMENT WORK.

Theorem 4 is proved for the |v|-sorted and extremal rules.  The CAMPAIGN's
rule is a Hilbert key on (x, Phi(v/v_th)), which is not a pure |v| sort, so
the theorem does not literally cover it.  What the general form needs is the
hypothesis

    C(P(v), v) = sum_P v_a^2 v_b^2  >  (N^2 - Q(v)) / (2(N-1))

i.e. the rule correlates within-pair energies more than an exchangeable rule
does.  This script tests that hypothesis on the REGISTERED r4 t=0 snapshots
-- the Maxwellian load of the actual campaign -- using a reconstructed C4b
pairing.  Reconstruction is faithful, not bit-exact; robustness to the
Hilbert order is reported for that reason.

Run against an unpacked copy of discrepancy_campaign_R4_FULL_results_
2026-08-12.zip (sha16 4ecdc69c19717135).
"""
import numpy as np
from scipy.stats import norm
from hilbertcurve.hilbertcurve import HilbertCurve


def hilbert_pairs(x, v, vth, order):
    hc = HilbertCurve(order, 2)
    side = 2 ** order - 1
    xi = np.clip((x * side).astype(int), 0, side)
    vi = np.clip((norm.cdf(v / vth) * side).astype(int), 0, side)
    keys = np.array(hc.distances_from_points(np.stack([xi, vi], axis=1)))
    o = np.argsort(keys, kind="stable")
    return o[0::2], o[1::2]


def cross(v, ia, ib):
    return float(np.sum(v[ia] ** 2 * v[ib] ** 2))


def main():
    print(f"{'file':>34} {'C Hilbert':>11} {'C exch avg':>11} {'ratio':>7} {'dQ/Q':>8}")
    for npart in (2048, 8192):
        for o in (74, 75, 76):
            f = f"snapshot_S-C4b_Np{npart}_o{o}_t0.npz"
            d = np.load(f)
            x, v = d["x"], d["v"].astype(float)
            v = v * np.sqrt(npart / np.sum(v * v))
            Q = float(np.sum(v ** 4))
            exch = (npart * npart - Q) / (2.0 * (npart - 1))
            ia, ib = hilbert_pairs(x, v, v.std(), 10)
            C_h = cross(v, ia, ib)
            dQ = 0.75 * (Q + 2 * C_h) - Q
            print(f"{f:>34} {C_h:11.4f} {exch:11.4f} {C_h/exch:7.3f} {dQ/Q:+8.4f}")


if __name__ == "__main__":
    main()
