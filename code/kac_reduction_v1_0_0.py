#!/usr/bin/env python3
"""
kac_reduction_v1_0_0.py
=======================
KAC-ONLY REDUCTION -- DISCLOSED DEVELOPMENT WORK (not a registered arm).

Purpose
-------
Strip the r3/r4 collisional PIC problem down to its Kac core: N particles,
one velocity coordinate each, no spatial dimension, no field solve, no
charge deposition, no grid.  Iterate pairing + energy-conserving rotation
and ask whether the VELOCITY CONDENSATE found in the r4 snapshots (excess
kurtosis +431.7, P(|v|>2 v_th) = 0.0063 vs Gaussian 0.0455, variance pinned
by the projection) reproduces in the reduction.

If it does, the mechanism is identified analytically and cheaply, and the
r5 point predictions acquire a mechanism instead of being bare bands.
If it does not, the field solve or the spatial pairing is implicated.

Design: the same 2x2 factorial that r5 registers.
    F1 pairing     : 'random' (uniform perfect matching)  vs 'sorted'
                     (adjacent ranks after sorting by v)
    F3 angle source: m-stratified sample with m = 1 (iid)  vs m = N_f
                     (one stratum per fired pair == the 1-d scrambled-net
                     stratification property, Ho & Owen 2026 Sec. 2.2)

    (random, iid)   ~ C3-analog      expected clean (classical Kac walk)
    (sorted, iid)   ~ A-iid-analog   UNTESTED in the record
    (random, LD)    ~ B-LD-analog    UNTESTED in the record
    (sorted, LD)    ~ C4b-analog     hypothesized to condense

Reduction of the pairing key: with no x coordinate the C4b Hilbert key on
(x, Phi(v/v_th)) degenerates to a sort on v, i.e. the C4a family.  r3 found
C4a and C4b statistically indistinguishable (both failed, 7/7 rungs), and
r4 reproduced that with M4a/M4b, so the reduction is faithful to both.

SEEDS: drawn from a namespace deliberately DISJOINT from the campaign
ledger (reduction base 813000000; campaign BASE is 20260803).  NO CAMPAIGN
OFFSET IS CONSUMED BY THIS SCRIPT.  Nothing here is registered; no verdict
of record may be built on it.

Units: v is dimensionless, in units of v_th (v_th = 1).  h_v = v_th/2 = 0.5
inherits the campaign's kernel width.  dt = 0.05, T = 40, nu*dt = 1 (every
pair fires each step), matching the campaign regime.
"""

import argparse
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------
# Constants inherited from the campaign regime (dimensionless)
# ----------------------------------------------------------------------
V_TH = 1.0            # velocity unit
H_V = 0.5             # kernel width = v_th/2, as in D_J's v-factor
DT = 0.05             # time step
T_END = 40.0          # end time
N_STEPS = int(round(T_END / DT))          # 800
SNAP_TIMES = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
SNAP_STEPS = tuple(int(round(t / DT)) for t in SNAP_TIMES)
WINDOW = (20.0, 40.0)                     # W*, inherited
RED_SEED_BASE = 813000000                 # disjoint from campaign BASE


# ----------------------------------------------------------------------
# Angle sampler: m-stratified on [0,1)
# ----------------------------------------------------------------------
def stratified_unit_sample(n_draws, m_strata, rng):
    """n_draws points on [0,1) from m equal strata, iid jitter within.

    m = 1        -> iid uniform (the negative control).
    m = n_draws  -> exactly one point per interval [j/n, (j+1)/n), which is
                    the 1-d nested-uniform scrambled-net law (the maximal
                    dose / positive control).
    Remainder rule (fixed here, matching the r5 draft Sec. 4.3): the first
    (n_draws mod m) strata each receive one extra draw.
    Assignment of the resulting values to pairs is a uniform random
    permutation, so no alignment between stratum index and pair rank
    survives -- consistent with r3 CTRL having exonerated the alignment
    channel.
    """
    m_strata = int(min(max(m_strata, 1), n_draws))
    counts = np.full(m_strata, n_draws // m_strata, dtype=np.int64)
    counts[: n_draws % m_strata] += 1
    stratum_of_draw = np.repeat(np.arange(m_strata, dtype=np.float64), counts)
    jitter = rng.random(n_draws)
    u = (stratum_of_draw + jitter) / m_strata
    rng.shuffle(u)
    return u


# ----------------------------------------------------------------------
# One Kac step
# ----------------------------------------------------------------------
def kac_step(v, pairing, m_strata, rng, fire_frac=1.0, project=True):
    """Advance the velocity vector by one collision step, in place-safe form.

    Rotation: for a fired pair (a, b) with R^2 = v_a^2 + v_b^2,
        v_a' = R cos(theta),  v_b' = R sin(theta),  theta = 2*pi*u.
    This is the full-circle rotation of r3 (r4 proved the monotone
    inverse-CDF variant has the identical joint law and identical results,
    so the reduction uses the original).  Pair energy is conserved exactly.

    project: mirror the campaign's exact global P/KE projection -- remove
    the mean, then rescale so sum(v^2) = N * v_th^2 exactly.  Kac rotation
    conserves energy but not momentum, so without the projection the mean
    performs a slow random walk.
    """
    n_part = v.size
    if pairing == "random":
        order = rng.permutation(n_part)
    elif pairing == "sorted":
        order = np.argsort(v, kind="stable")
    else:
        raise ValueError("pairing must be 'random' or 'sorted'")

    idx_a = order[0::2]
    idx_b = order[1::2]
    n_pairs = idx_a.size

    if fire_frac >= 1.0:
        fired = np.arange(n_pairs)
    else:
        n_fire = int(round(fire_frac * n_pairs))
        fired = rng.choice(n_pairs, size=n_fire, replace=False)
    idx_a = idx_a[fired]
    idx_b = idx_b[fired]
    n_fired = idx_a.size

    m_eff = n_fired if m_strata == 0 else m_strata      # 0 is the sentinel for "maximal"
    u = stratified_unit_sample(n_fired, m_eff, rng)
    theta = 2.0 * np.pi * u

    radius = np.hypot(v[idx_a], v[idx_b])
    v[idx_a] = radius * np.cos(theta)
    v[idx_b] = radius * np.sin(theta)

    if project:
        v -= v.mean()
        energy = np.dot(v, v)
        if energy > 0.0:
            v *= np.sqrt(n_part * V_TH * V_TH / energy)
    return v


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def mmd_to_maxwellian(v, h=H_V, sigma=V_TH, chunk=2048):
    """D_v = sqrt(MMD^2) between the empirical law of v and N(0, sigma^2),
    Gaussian kernel k(u,w) = exp(-(u-w)^2 / (2 h^2)).

    Closed forms for the cross and target terms:
        E_{w~p} k(v, w)      = (h / sqrt(h^2+s^2)) exp(-v^2 / (2(h^2+s^2)))
        E_{w,w'~p} k(w, w')  =  h / sqrt(h^2 + 2 s^2)
    The empirical term is the V-statistic (diagonal included), matching the
    campaign's convention that the iid plateau is a positive constant to be
    divided out rather than a debiased zero.
    """
    n_part = v.size
    h2 = h * h
    s2 = sigma * sigma

    total = 0.0
    for start in range(0, n_part, chunk):
        block = v[start : start + chunk]
        diff = block[:, None] - v[None, :]
        total += float(np.exp(-(diff * diff) / (2.0 * h2)).sum())
    term_emp = total / (n_part * n_part)

    term_cross = 2.0 * float(
        np.mean((h / np.sqrt(h2 + s2)) * np.exp(-(v * v) / (2.0 * (h2 + s2))))
    )
    term_tgt = h / np.sqrt(h2 + 2.0 * s2)

    mmd_sq = term_emp - term_cross + term_tgt
    return float(np.sqrt(max(mmd_sq, 0.0)))


def dv_plateau(n_part):
    """Analytic iid plateau of D_v at this N.

    The centered kernel k~(u,w) = k(u,w) - mu(u) - mu(w) + E_pp is degenerate
    under the target, so the V-statistic's leading term is the diagonal:
        E[D_v^2] = (1/N) E[k~(v,v)] = (1 - E_pp)/N,
        E_pp = h / sqrt(h^2 + 2 s^2) = 0.5/1.5 = 1/3  for h = v_th/2,
    giving plateau^2 = (2/3)/N -- IDENTICALLY the campaign's registered
    analytic plateau sqrt((2/3)/Np).  That coincidence is not cosmetic: it
    confirms the reduction's D_v is the same object as the campaign's, so
    the ratios below are directly comparable to S-C4b's 35.0 and 72.3.

    Using the analytic form rather than an empirical estimate also removes
    a real statistical trap: D_v^2 is a degenerate V-statistic whose
    relative spread is O(1) regardless of N, so an empirical plateau from a
    handful of draws is far too noisy to serve as a denominator.
    """
    return float(np.sqrt((1.0 - H_V / np.sqrt(H_V * H_V + 2.0 * V_TH * V_TH)) / n_part))


def state_diagnostics(v):
    """Condensate observables, matching what the r4 snapshot analysis
    reported: variance (pinned by the projection), excess kurtosis, and the
    two-sigma tail mass (Gaussian reference 0.0455)."""
    mean_v = float(v.mean())
    centered = v - mean_v
    var = float(np.mean(centered * centered))
    if var > 0.0:
        kurt_excess = float(np.mean(centered ** 4) / (var * var) - 3.0)
    else:
        kurt_excess = float("nan")
    tail = float(np.mean(np.abs(v) > 2.0 * V_TH))
    return {
        "mean_v": mean_v,
        "var_over_vth2": var / (V_TH * V_TH),
        "excess_kurtosis": kurt_excess,
        "tail_frac_2vth": tail,
    }


# ----------------------------------------------------------------------
# One replicate
# ----------------------------------------------------------------------
def run_replicate(n_part, pairing, m_strata, seed, fire_frac=1.0, project=True,
                  with_dv=True, snap_steps=SNAP_STEPS):
    """Run one replicate; return a list of per-snapshot diagnostic rows."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, V_TH, size=n_part)
    if project:
        v -= v.mean()
        v *= np.sqrt(n_part * V_TH * V_TH / np.dot(v, v))

    plateau = dv_plateau(n_part)

    rows = []
    snap_set = set(snap_steps)

    def record(step_idx):
        diag = state_diagnostics(v)
        d_v = mmd_to_maxwellian(v) if with_dv else np.nan
        rows.append(
            dict(
                n_part=n_part,
                pairing=pairing,
                m_strata=("N_f" if m_strata == 0 else m_strata),
                seed=seed,
                step=step_idx,
                t=step_idx * DT,
                d_v=d_v,
                d_v_plateau=plateau,
                d_v_ratio=(d_v / plateau) if with_dv else np.nan,
                **diag,
            )
        )

    if 0 in snap_set:
        record(0)
    for step_idx in range(1, N_STEPS + 1):
        kac_step(v, pairing, m_strata, rng, fire_frac=fire_frac, project=project)
        if step_idx in snap_set:
            record(step_idx)
    return rows


# ----------------------------------------------------------------------
# Phase A: verification of the reduction itself
# ----------------------------------------------------------------------
def phase_verify():
    print("=" * 70)
    print("KAC REDUCTION -- VERIFICATION (R-V1 .. R-V4)")
    print("=" * 70)
    ok = True
    rng = np.random.default_rng(RED_SEED_BASE)

    # R-V1: pair energy conserved exactly by the rotation
    v = rng.normal(0, 1, 4096)
    e0 = float(np.dot(v, v))
    v2 = v.copy()
    kac_step(v2, "sorted", 0, np.random.default_rng(1), project=False)
    rel = abs(float(np.dot(v2, v2)) - e0) / e0
    passed = rel < 1e-12
    ok &= passed
    print(f"R-V1 {'PASS' if passed else 'FAIL'}  rotation conserves energy: rel {rel:.2e} (< 1e-12)")

    # R-V2: classical Kac walk (random pairing, iid angles) stays Maxwellian
    rows = run_replicate(4096, "random", 1, RED_SEED_BASE + 1, with_dv=False)
    kurt_end = rows[-1]["excess_kurtosis"]
    passed = abs(kurt_end) < 0.30
    ok &= passed
    print(f"R-V2 {'PASS' if passed else 'FAIL'}  classical Kac stays Gaussian: "
          f"excess kurtosis at t=40 = {kurt_end:+.4f} (|.| < 0.30)")

    # R-V3: maximal-dose sampler has the 1-d net stratification property
    n = 1024
    u = stratified_unit_sample(n, n, np.random.default_rng(3))
    counts = np.bincount(np.floor(u * n).astype(int), minlength=n)
    passed = bool(np.all(counts == 1))
    ok &= passed
    print(f"R-V3 {'PASS' if passed else 'FAIL'}  m=N_f sampler: exactly one point "
          f"per 1/N interval (min {counts.min()}, max {counts.max()})")

    # R-V4: the analytic plateau is the mean of D_v^2 over iid Maxwellian
    # draws.  D_v^2 is a degenerate V-statistic (relative spread O(1)), so
    # this is tested on the MEAN of D_v^2 over many draws, not on a few.
    for n_test, n_draws, tol in ((512, 400, 0.10), (2048, 200, 0.12)):
        g = np.random.default_rng(4000 + n_test)
        vals = [mmd_to_maxwellian(g.normal(0.0, V_TH, size=n_test)) ** 2
                for _ in range(n_draws)]
        got = float(np.mean(vals))
        want = dv_plateau(n_test) ** 2
        rel = abs(got - want) / want
        passed = rel < tol
        ok &= passed
        print(f"R-V4 {'PASS' if passed else 'FAIL'}  N={n_test:5d}: mean D_v^2 = {got:.6e} "
              f"vs analytic (2/3)/N = {want:.6e}  (rel {rel:.3f} < {tol})")

    print("-" * 70)
    print(f"{'ALL VERIFICATION TESTS PASSED.' if ok else 'VERIFICATION FAILED.'}")
    print("=" * 70)
    return ok


# ----------------------------------------------------------------------
# Phase B: the 2x2 factorial
# ----------------------------------------------------------------------
def phase_factorial(out_csv, n_parts=(8192,), n_reps=8):
    cells = [
        ("random", 1, "C3-analog      (random pairing, iid angles)"),
        ("sorted", 1, "A-iid-analog   (sorted pairing, iid angles)"),
        ("random", 0, "B-LD-analog    (random pairing, LD multiset)"),
        ("sorted", 0, "C4b-analog     (sorted pairing, LD multiset)"),
    ]
    rows = []
    t_start = time.time()
    for n_part in n_parts:
        for ci, (pairing, m_strata, label) in enumerate(cells):
            for rep in range(n_reps):
                seed = RED_SEED_BASE + 1000 * ci + 10 * rep + (1 if n_part == 8192 else 0)
                rows.extend(run_replicate(n_part, pairing, m_strata, seed))
                _flush(rows, out_csv)
            sub = [r for r in rows if r["n_part"] == n_part
                   and r["pairing"] == pairing
                   and r["m_strata"] == ("N_f" if m_strata == 0 else m_strata)
                   and r["t"] == 40.0]
            kurt = np.mean([r["excess_kurtosis"] for r in sub])
            tail = np.mean([r["tail_frac_2vth"] for r in sub])
            ratio = np.exp(np.mean(np.log([r["d_v_ratio"] for r in sub])))
            print(f"  Np={n_part:5d}  {label}  t=40: "
                  f"kurt {kurt:+9.2f}   tail {tail:.4f}   D_v/plateau {ratio:8.2f}")
    print(f"  [phase B wall clock {time.time() - t_start:.1f}s]")
    return rows


# ----------------------------------------------------------------------
# Phase C: dose response
# ----------------------------------------------------------------------
def phase_dose(out_csv, n_parts=(2048,), n_reps=4,
               doses=(1, 2, 8, 32, 128, 0)):
    rows = []
    t_start = time.time()
    for n_part in n_parts:
        for di, m_strata in enumerate(doses):
            for rep in range(n_reps):
                seed = RED_SEED_BASE + 500000 + 1000 * di + 10 * rep + (1 if n_part == 8192 else 0)
                rows.extend(run_replicate(n_part, "sorted", m_strata, seed,
                                          snap_steps=(0, 400, 800)))
                _flush(rows, out_csv)
            label = "N_f" if m_strata == 0 else str(m_strata)
            sub = [r for r in rows if r["n_part"] == n_part
                   and r["m_strata"] == ("N_f" if m_strata == 0 else m_strata)
                   and r["t"] == 40.0]
            kurt = np.mean([r["excess_kurtosis"] for r in sub])
            ratio = np.exp(np.mean(np.log([r["d_v_ratio"] for r in sub])))
            print(f"  Np={n_part:5d}  m={label:>5s}  t=40: "
                  f"kurt {kurt:+9.2f}   D_v/plateau {ratio:8.2f}")
    print(f"  [phase C wall clock {time.time() - t_start:.1f}s]")
    return rows


# ----------------------------------------------------------------------
def _flush(rows, path):
    """Crash-safe incremental write."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["verify", "factorial", "dose"], default="verify")
    ap.add_argument("--out", default=".")
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()

    if args.phase == "verify":
        sys.exit(0 if phase_verify() else 1)
    elif args.phase == "factorial":
        out = os.path.join(args.out, "kac_reduction_factorial_Np8192.csv")
        phase_factorial(out, n_reps=args.reps or 8)
        print(f"wrote {out}")
    else:
        out = os.path.join(args.out, "kac_reduction_dose.csv")
        phase_dose(out, n_reps=args.reps or 4)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
