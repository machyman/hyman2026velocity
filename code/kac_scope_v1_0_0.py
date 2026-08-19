#!/usr/bin/env python3
"""
kac_scope_v1_0_0.py -- DISCLOSED DEVELOPMENT WORK. Zero campaign offsets.

The two conjectures the outline flagged as needing measurement before prose:

C1  SPATIAL-CELL SAFETY.  Binary-collision schemes of Takizuka-Abe type sort
    into spatial cells and pair at random inside each.  At equilibrium the
    velocities are independent of position, so such a rule should be
    exchangeable in v and give C/C-bar = 1.  Tested on the SAME registered
    t=0 snapshots used for the Hilbert transfer test, so the two numbers are
    directly comparable.

C4  REALISTIC SCATTERING KERNELS.  The strongest referee objection is that
    the Kac step resamples the collision angle uniformly, whereas Coulomb
    collisions are small-angle.  Replace "resample the angle" by "rotate the
    pair by a small angle theta":

        v_a' = v_a cos t - v_b sin t,   v_b' = v_a sin t + v_b cos t.

    Writing (v_a, v_b) = R(cos phi, sin phi), one has
        v_a^4 + v_b^4 = R^4 (3/4 + (1/4) cos 4phi),
    so for theta symmetric about 0 with kappa = E[cos 4theta],

        E_theta[Q(Tv) | v] = kappa * Q + (1 - kappa) * (3/4)(Q + 2C).   (*)

    kappa = 0 recovers Lemma 2 (uniform resampling).  kappa -> 1 is the
    no-collision limit.  For ANY kappa < 1 the pairing still enters only
    through C, and the drift away from equilibrium is just attenuated by
    (1 - kappa).  If (*) holds numerically, the obstruction extends to every
    symmetric scattering law and the objection is answered analytically
    rather than by a sweep.
"""

import numpy as np
from scipy.stats import norm
from hilbertcurve.hilbertcurve import HilbertCurve


# ----------------------------------------------------------------- C1
def c_over_cbar(v, idx_a, idx_b):
    n = v.size
    Q = float(np.sum(v ** 4))
    C = float(np.sum(v[idx_a] ** 2 * v[idx_b] ** 2))
    return C / ((n * n - Q) / (2.0 * (n - 1)))


def spatial_cell_pairs(x, n_cells, rng):
    """Sort into n_cells equal spatial cells; uniform random matching inside
    each.  Leftover odd particles within a cell are carried to a pooled
    remainder and matched at random there."""
    n = x.size
    cell = np.clip((x * n_cells).astype(int), 0, n_cells - 1)
    a_list, b_list, leftover = [], [], []
    for c in range(n_cells):
        members = np.flatnonzero(cell == c)
        if members.size < 2:
            leftover.extend(members.tolist())
            continue
        members = rng.permutation(members)
        m = members.size // 2 * 2
        a_list.append(members[:m:2])
        b_list.append(members[1:m:2])
        if members.size % 2:
            leftover.append(int(members[-1]))
    if len(leftover) >= 2:
        lo = rng.permutation(np.array(leftover))
        m = lo.size // 2 * 2
        a_list.append(lo[:m:2])
        b_list.append(lo[1:m:2])
    return np.concatenate(a_list), np.concatenate(b_list)


def hilbert_pairs(x, v, vth, order=10):
    hc = HilbertCurve(order, 2)
    side = 2 ** order - 1
    xi = np.clip((x * side).astype(int), 0, side)
    vi = np.clip((norm.cdf(v / vth) * side).astype(int), 0, side)
    keys = np.array(hc.distances_from_points(np.stack([xi, vi], axis=1)))
    o = np.argsort(keys, kind="stable")
    return o[0::2], o[1::2]


def run_C1():
    print("=" * 72)
    print("C1  SPATIAL-CELL PAIRING vs HILBERT SORT, on the registered t=0 loads")
    print("=" * 72)
    print(f"{'Np':>6} {'cells':>6} {'C/C-bar spatial':>17} {'C/C-bar Hilbert':>17}")
    for npart in (2048, 8192):
        for o in (74, 75, 76):
            d = np.load(f"snapshot_S-C4b_Np{npart}_o{o}_t0.npz")
            x, v = d["x"], d["v"].astype(float)
            v = v * np.sqrt(npart / np.sum(v * v))
            rng = np.random.default_rng(813000000 + 300000 + o)
            sa, sb = spatial_cell_pairs(x, 64, rng)
            ha, hb = hilbert_pairs(x, v, v.std())
            print(f"{npart:>6} {64:>6} {c_over_cbar(v, sa, sb):>17.4f} "
                  f"{c_over_cbar(v, ha, hb):>17.4f}")
            break   # the t=0 load is common to offsets 74-76
    print("\n  Prediction: spatial ~ 1 (exchangeable in v), Hilbert ~ 2.7-2.8.")


# ----------------------------------------------------------------- C4
def run_C4():
    print("\n" + "=" * 72)
    print("C4  SCATTERING KERNEL: does the identity survive small-angle collisions?")
    print("=" * 72)
    rng = np.random.default_rng(813000000 + 400000)
    n = 2048
    v = rng.normal(size=n)
    v *= np.sqrt(n / np.dot(v, v))
    order = np.argsort(np.abs(v), kind="stable")
    ia, ib = order[0::2], order[1::2]          # sorted rule
    Q = float(np.sum(v ** 4))
    C = float(np.sum(v[ia] ** 2 * v[ib] ** 2))

    print(f"{'delta':>10} {'kappa=E[cos4t]':>16} {'simulated E[Q_after]':>22} "
          f"{'identity (*)':>16} {'rel':>10}")
    for delta in (np.pi, np.pi / 2, np.pi / 8, np.pi / 16, 0.1):
        kappa = np.sin(4 * delta) / (4 * delta) if delta > 0 else 1.0
        sims = []
        for _ in range(400):
            t = rng.uniform(-delta, delta, size=ia.size)
            va, vb = v[ia], v[ib]
            na = va * np.cos(t) - vb * np.sin(t)
            nb = va * np.sin(t) + vb * np.cos(t)
            w = v.copy()
            w[ia], w[ib] = na, nb
            sims.append(float(np.sum(w ** 4)))
        sim = float(np.mean(sims))
        ident = kappa * Q + (1 - kappa) * 0.75 * (Q + 2 * C)
        print(f"{delta:>10.4f} {kappa:>16.4f} {sim:>22.6f} {ident:>16.6f} "
              f"{abs(sim-ident)/ident:>10.2e}")

    print("\n  If the identity column tracks the simulated column, the pairing")
    print("  enters only through C for EVERY symmetric scattering law, and the")
    print("  drift is attenuated by (1 - kappa) but never removed.")

    # does the condensate still form under small-angle scattering?
    print("\n  Condensate under small-angle scattering (sorted vs random pairing):")
    print(f"{'delta':>10} {'kappa':>8} {'steps':>8} {'m4 sorted':>12} {'m4 random':>12}")
    for delta in (np.pi, np.pi / 8, np.pi / 16):
        kappa = np.sin(4 * delta) / (4 * delta)
        steps = int(round(800 / max(1 - kappa, 1e-3)))
        steps = min(steps, 12000)
        out = {}
        for rule in ("sorted", "random"):
            g = np.random.default_rng(813000000 + 410000 + int(delta * 1000))
            w = g.normal(size=n)
            w -= w.mean(); w *= np.sqrt(n / np.dot(w, w))
            for _ in range(steps):
                if rule == "sorted":
                    o = np.argsort(np.abs(w), kind="stable")
                else:
                    o = g.permutation(n)
                a, b = o[0::2], o[1::2]
                t = g.uniform(-delta, delta, size=a.size)
                wa, wb = w[a], w[b]
                w[a] = wa * np.cos(t) - wb * np.sin(t)
                w[b] = wa * np.sin(t) + wb * np.cos(t)
                w -= w.mean(); w *= np.sqrt(n / np.dot(w, w))
            out[rule] = float(np.mean(w ** 4)) / 3.0
        print(f"{delta:>10.4f} {kappa:>8.4f} {steps:>8d} {out['sorted']:>12.3f} "
              f"{out['random']:>12.3f}")


if __name__ == "__main__":
    run_C1()
    run_C4()
