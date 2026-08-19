#!/usr/bin/env python3
"""
kac_reduction_v1_1_0.py
=======================
KAC-ONLY REDUCTION, PHASE D: MECHANISM DISSECTION.
DISCLOSED DEVELOPMENT WORK -- not a registered arm, zero campaign offsets
(reduction seed base 813000000, disjoint from campaign BASE 20260803).

v1_0_0 established that the condensate is caused by rank-based PAIRING, not
by the low-discrepancy angle multiset, and that it reproduces the registered
S-C4b D_v ratios to 0.6-1.7%.  v1_1_0 asks WHICH property of the pairing
does it, by discriminating three accounts that all predict "sorted pairing
condenses":

  A1 RATCHET / absorbing region.  Sorting pairs near-equal velocities; a
     pair at (v,v) lands at (R cos t, R sin t), so angles near an axis strand
     one member at ~0.  Sorting then re-pairs near-zero particles WITH EACH
     OTHER, where R ~ 0 and they cannot recover, while energetic particles
     keep splitting.  Requires SIMILARITY of the paired velocities.

  A2 DETERMINISTIC-MAP FIXED POINT.  Sorting makes the map on the empirical
     measure nearly deterministic, and its fixed point need not be Maxwellian.
     Requires STRUCTURE, not similarity.

  A3 EXCHANGEABILITY BREAKING.  Any state-dependent pairing breaks the Kac
     process's reversibility w.r.t. the uniform-on-sphere measure, so ANY
     such rule distorts.  Requires only STATE DEPENDENCE.

Discriminator: ANTI-SORTED pairing (rank i with rank N+1-i) is maximally
state-dependent and maximally structured, but pairs the most DISSIMILAR
velocities.  A1 predicts it does NOT condense.  A2 predicts some other
non-Maxwellian fixed point.  A3 predicts it distorts too.

Locality knob: 'block:w' sorts by v, cuts the sorted list into blocks of w
consecutive ranks, and draws a uniform random matching inside each block.
  w = 2   -> strict rank adjacency (== 'sorted')
  w = Np  -> a single block == global uniform random matching (== 'random')
so w interpolates continuously between the two cells of the v1_0_0
factorial, with no other change.  This is the dose-response the mechanism
predicts, in place of the angle-multiset dose that v1_0_0 found flat.

Controls a referee will demand:
  - projection OFF: does the global P/KE projection CREATE the condensate?
  - firing fraction < 1: is saturation (nu*dt = 1, every pair firing) needed?
  - |v|-sort vs v-sort: is it similarity in speed or in signed velocity?
"""

import argparse
import time

import numpy as np

from kac_reduction_v1_0_0 import (
    V_TH, H_V, DT, N_STEPS, RED_SEED_BASE,
    stratified_unit_sample, mmd_to_maxwellian, dv_plateau, state_diagnostics,
)

SNAP_STEPS_D = (0, 400, 800)          # t = 0, 20, 40


# ----------------------------------------------------------------------
def make_pairs(v, pairing, rng):
    """Return (idx_a, idx_b) for one step under the named pairing rule."""
    n_part = v.size

    if pairing == "random":
        order = rng.permutation(n_part)
        return order[0::2], order[1::2]

    if pairing == "sorted":
        order = np.argsort(v, kind="stable")
        return order[0::2], order[1::2]

    if pairing == "abssorted":
        order = np.argsort(np.abs(v), kind="stable")
        return order[0::2], order[1::2]

    if pairing == "absanti":
        # sort by SPEED, pair fastest with slowest: maximal within-pair
        # energy DIFFERENCE.  Maximally state-dependent, minimally similar.
        order = np.argsort(np.abs(v), kind="stable")
        half = n_part // 2
        return order[:half], order[::-1][:half]

    if pairing == "antisorted":
        order = np.argsort(v, kind="stable")
        half = n_part // 2
        return order[:half], order[::-1][:half]

    if pairing.startswith("block:"):
        width = int(pairing.split(":")[1])
        width = int(min(max(width, 2), n_part))
        order = np.argsort(v, kind="stable")
        n_blocks = n_part // width
        head = order[: n_blocks * width].reshape(n_blocks, width)
        shuffled = np.take_along_axis(
            head, np.argsort(rng.random(head.shape), axis=1), axis=1
        ).ravel()
        tail = order[n_blocks * width:]
        if tail.size:
            tail = rng.permutation(tail)
            shuffled = np.concatenate([shuffled, tail])
        return shuffled[0::2], shuffled[1::2]

    raise ValueError(f"unknown pairing {pairing!r}")


def kac_step_d(v, pairing, m_strata, rng, fire_frac=1.0, project=True):
    n_part = v.size
    idx_a, idx_b = make_pairs(v, pairing, rng)
    n_pairs = idx_a.size

    if fire_frac < 1.0:
        n_fire = int(round(fire_frac * n_pairs))
        sel = rng.choice(n_pairs, size=n_fire, replace=False)
        idx_a, idx_b = idx_a[sel], idx_b[sel]
    n_fired = idx_a.size

    m_eff = n_fired if m_strata == 0 else m_strata
    theta = 2.0 * np.pi * stratified_unit_sample(n_fired, m_eff, rng)
    radius = np.hypot(v[idx_a], v[idx_b])
    v[idx_a] = radius * np.cos(theta)
    v[idx_b] = radius * np.sin(theta)

    if project:
        v -= v.mean()
        energy = np.dot(v, v)
        if energy > 0.0:
            v *= np.sqrt(n_part * V_TH * V_TH / energy)
    return v


def run_rep_d(n_part, pairing, m_strata, seed, fire_frac=1.0, project=True):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, V_TH, size=n_part)
    v -= v.mean()
    v *= np.sqrt(n_part * V_TH * V_TH / np.dot(v, v))
    plateau = dv_plateau(n_part)
    rows = []
    for step_idx in range(N_STEPS + 1):
        if step_idx:
            kac_step_d(v, pairing, m_strata, rng, fire_frac, project)
        if step_idx in SNAP_STEPS_D:
            d_v = mmd_to_maxwellian(v)
            rows.append(dict(n_part=n_part, pairing=pairing,
                             m_strata=("N_f" if m_strata == 0 else m_strata),
                             fire_frac=fire_frac, project=int(project), seed=seed,
                             t=step_idx * DT, d_v=d_v, d_v_ratio=d_v / plateau,
                             **state_diagnostics(v)))
    return rows


def summarize(rows, n_part, key, val):
    sub = [r for r in rows if r["n_part"] == n_part and r[key] == val and r["t"] == 40.0]
    if not sub:
        return None
    return (np.exp(np.mean(np.log([r["d_v_ratio"] for r in sub]))),
            float(np.mean([r["excess_kurtosis"] for r in sub])),
            float(np.mean([r["tail_frac_2vth"] for r in sub])))


def flush(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["discrim", "locality", "controls"], default="discrim")
    ap.add_argument("--np", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    n_part, n_reps = args.np, args.reps
    t0 = time.time()

    if args.block == "discrim":
        print(f"=== D1 MECHANISM DISCRIMINATION (Np={n_part}, {n_reps} reps, iid angles) ===")
        rows = []
        for pi, pairing in enumerate(["random", "sorted", "antisorted", "abssorted", "absanti"]):
            for rep in range(n_reps):
                rows.extend(run_rep_d(n_part, pairing, 1, RED_SEED_BASE + 900000 + 1000 * pi + 10 * rep))
            dv, ku, tl = summarize(rows, n_part, "pairing", pairing)
            print(f"  {pairing:12s}  D_v/plateau {dv:8.2f}   kurt {ku:+10.2f}   tail {tl:.4f}")
        flush(rows, f"kac_dissect_discrim_Np{n_part}.csv")

    elif args.block == "locality":
        widths = [2, 4, 8, 16, 64, 256, 1024, n_part]
        print(f"=== D2 LOCALITY DOSE (Np={n_part}, {n_reps} reps, iid angles) ===")
        print("    w = 2 is strict rank adjacency; w = Np is global random matching")
        rows = []
        for wi, width in enumerate(widths):
            pairing = f"block:{width}"
            for rep in range(n_reps):
                rows.extend(run_rep_d(n_part, pairing, 1, RED_SEED_BASE + 700000 + 1000 * wi + 10 * rep))
            dv, ku, tl = summarize(rows, n_part, "pairing", pairing)
            print(f"  w={width:6d}  D_v/plateau {dv:8.2f}   kurt {ku:+10.2f}   tail {tl:.4f}")
        flush(rows, f"kac_dissect_locality_Np{n_part}.csv")

    else:
        print(f"=== D3/D4 CONTROLS (Np={n_part}, {n_reps} reps, sorted pairing, iid angles) ===")
        rows = []
        for ci, (proj, frac, label) in enumerate([
            (True, 1.0, "projection ON,  fire 1.00  (reference)"),
            (False, 1.0, "projection OFF, fire 1.00"),
            (True, 0.50, "projection ON,  fire 0.50"),
            (True, 0.25, "projection ON,  fire 0.25"),
        ]):
            tag = []
            for rep in range(n_reps):
                r = run_rep_d(n_part, "sorted", 1, RED_SEED_BASE + 600000 + 1000 * ci + 10 * rep,
                              fire_frac=frac, project=proj)
                rows.extend(r)
                tag.extend([x for x in r if x["t"] == 40.0])
            dv = np.exp(np.mean(np.log([r["d_v_ratio"] for r in tag])))
            ku = np.mean([r["excess_kurtosis"] for r in tag])
            va = np.mean([r["var_over_vth2"] for r in tag])
            print(f"  {label:38s}  D_v/plateau {dv:8.2f}   kurt {ku:+10.2f}   var {va:.4f}")
        flush(rows, f"kac_dissect_controls_Np{n_part}.csv")

    print(f"  [wall clock {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
