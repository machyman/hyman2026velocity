#!/usr/bin/env python3
"""Test the monotone energy cascade of the sorted rule.

THE CLAIM (Theorem, Section 9 of hyman2026velocity).  Let N be even, let the
sorted rule order the particles by |v| and pair adjacent ranks, and for even
j let E_j be the sum of the j smallest squared velocities, the energy held by
the j slowest particles.  Then E_j is non-increasing along every trajectory.

THE ARGUMENT.  The j slowest, j even, occupy exactly j/2 complete pairs.
Each collision rotates its own pair on the circle of radius sqrt(v_a^2+v_b^2)
and so conserves that pair's energy, so the same j particles still hold
exactly E_j after the step.  The j smallest post-step squares sum to at most
the sum over any j particles, these included.  Hence E_j' <= E_j.  For odd j
the argument fails: rank j is paired with rank j+1 across the boundary, and
E_j can rise.  That failure is the control.

WHAT THIS SCRIPT MEASURES.  The theorem is a one-step, pointwise statement,
and pointwise monotonicity at every state gives every trajectory by
induction.  So the primary test is ONE STEP from many states drawn to cover
the space, not a long trajectory, which would spend nearly all its time near
the condensate where E_j is already small.

  Part A, one step.  For each N and each seed, states are drawn from five
  families and stepped once: uniform on the sphere; two-particle condensate
  with the remainder holding 1e-2 or 1e-6 of the energy; every speed tied;
  speeds tied in random index pairs; and magnitudes log-uniform over eight
  decades.  For every j, even and odd, the maximum of E_j' - E_j over all
  states is recorded, together with P(E_j' < E_j) and the mean relative
  decrease.  The last two are data for the remaining conjecture, that the
  limits are zero, which needs lower bounds on exactly those quantities.

  Part B, trajectories.  200 chains, 300 steps, from uniform starts.  The
  maximum one-step increase of E_j over every chain and every step.

  Self-check.  j = N is the total energy and must be conserved to rounding
  under any rule; it is reported and it must sit at the rounding floor.

THE GATE.  A correct implementation observes E_j' - E_j at the rounding
floor, about one unit in the last place of N, but the rigorous bound for
rotating and summing j nonnegative doubles is of order N^2 eps.  PASS for
even j requires max(E_j' - E_j) <= TOL_FACTOR * N^2 * eps.  The observed
maximum is reported in units of one ulp of N so the reader sees how far
below the gate it sits.  The odd-j control is expected to fail by many
orders of magnitude; its value is reported, not gated.

step_sphere is reproduced verbatim from kac_manuscript_data_v1_0_0.py so
this check runs on the same implementation of the dynamics that produced
every figure and table in the paper.

Input   none (self-contained simulation)
Output  cascade_monotone.csv

Determinism.  Seeded from SEEDS below; reproducible on any machine with the
same numpy.  No figure, no timestamp.  Run twice and diff: the CSV must be
byte-identical.

Usage:  python3 kac_cascade_monotone_v1_0_0.py [--outdir DIR] [--quick]
"""
import argparse
import csv
import hashlib
import os
import sys

import numpy as np

N_VALUES = (4, 8, 16, 64, 256)
SEEDS = (813950021, 813950022)
STATES_PER_FAMILY = 20000       # Part A, per family, per seed, per N
TRAJ_CHAINS, TRAJ_STEPS = 200, 300
TOL_FACTOR = 4.0
EPS = np.finfo(np.float64).eps


# --------------------------------------------------------------- dynamics
# Reproduced verbatim from kac_manuscript_data_v1_0_0.py.  Do not edit.
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


# ------------------------------------------------------------ observables
def prefix_energy(v):
    """E_j for j = 1..N, per chain: cumulative sum of the sorted squares."""
    return np.cumsum(np.sort(v * v, axis=1), axis=1)


def to_sphere(v, n_part):
    return v * np.sqrt(n_part / np.sum(v * v, axis=1, keepdims=True))


# --------------------------------------------------------- state families
def fam_uniform(m, n, rng):
    return to_sphere(rng.normal(size=(m, n)), n)


def fam_condensate(m, n, rng, eps_frac):
    v = rng.normal(size=(m, n))
    v = to_sphere(v, n)
    # two random particles take (1 - eps_frac) of the energy, split at a
    # uniform phase; the other n-2 keep their shape but hold eps_frac
    idx = np.array([rng.choice(n, 2, replace=False) for _ in range(m)])
    big = np.zeros((m, n))
    rest = v.copy()
    rest[np.arange(m)[:, None], idx] = 0.0
    rest *= np.sqrt(eps_frac * n / np.sum(rest * rest, axis=1, keepdims=True))
    phase = 2.0 * np.pi * rng.random(m)
    rad = np.sqrt((1.0 - eps_frac) * n)
    big[np.arange(m), idx[:, 0]] = rad * np.cos(phase)
    big[np.arange(m), idx[:, 1]] = rad * np.sin(phase)
    return big + rest


def fam_ties_all(m, n, rng):
    """Every speed equal: |v_i| = 1 exactly, random signs.  E_j = j exactly."""
    return np.where(rng.random((m, n)) < 0.5, -1.0, 1.0)


def fam_ties_pairs(m, n, rng):
    """Speeds tied in n/2 random index pairs, so the stable sort must break
    every tie; the tie-break is not the sorted rule's own pairing."""
    v = np.abs(rng.normal(size=(m, n // 2)))
    v = np.repeat(v, 2, axis=1)
    perm = np.array([rng.permutation(n) for _ in range(m)])
    v = np.take_along_axis(v, perm, axis=1)
    v *= np.where(rng.random((m, n)) < 0.5, -1.0, 1.0)
    return to_sphere(v, n)


def fam_spread(m, n, rng):
    """Magnitudes log-uniform over eight decades, then normalised."""
    mag = 10.0 ** rng.uniform(-8.0, 0.0, size=(m, n))
    sgn = np.where(rng.random((m, n)) < 0.5, -1.0, 1.0)
    return to_sphere(mag * sgn, n)


FAMILIES = (
    ("uniform",        lambda m, n, r: fam_uniform(m, n, r)),
    ("condensate_1e2", lambda m, n, r: fam_condensate(m, n, r, 1e-2)),
    ("condensate_1e6", lambda m, n, r: fam_condensate(m, n, r, 1e-6)),
    ("ties_all",       lambda m, n, r: fam_ties_all(m, n, r)),
    ("ties_pairs",     lambda m, n, r: fam_ties_pairs(m, n, r)),
    ("spread",         lambda m, n, r: fam_spread(m, n, r)),
)


# ------------------------------------------------------------------ parts
def one_step_stats(before, after):
    """Per j: max increase, P(strict decrease), mean relative decrease."""
    d = after - before                                    # (m, N)
    max_inc = d.max(axis=0)
    p_dec = (d < 0.0).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(before > 0.0, -d / before, 0.0)
    mean_rel = rel.mean(axis=0)
    return max_inc, p_dec, mean_rel


def part_a(n, seed, m):
    rng = np.random.default_rng(seed)
    out = []
    for name, make in FAMILIES:
        v = make(m, n, rng)
        before = prefix_energy(v)
        step_sphere(v, "sorted", rng)
        after = prefix_energy(v)
        mi, pd, mr = one_step_stats(before, after)
        out.append((name, m, mi, pd, mr))
    return out


def part_b(n, seed, chains, steps):
    rng = np.random.default_rng(seed + 1000)
    v = fam_uniform(chains, n, rng)
    before = prefix_energy(v)
    max_inc = np.full(n, -np.inf)
    n_dec = np.zeros(n)
    rel_acc = np.zeros(n)
    for _ in range(steps):
        step_sphere(v, "sorted", rng)
        after = prefix_energy(v)
        d = after - before
        max_inc = np.maximum(max_inc, d.max(axis=0))
        n_dec += (d < 0.0).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_acc += np.where(before > 0.0, -d / before, 0.0).sum(axis=0)
        before = after
    tot = chains * steps
    return ("trajectory", tot, max_inc, n_dec / tot, rel_acc / tot)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    m = 2000 if args.quick else STATES_PER_FAMILY
    tc, ts = (50, 60) if args.quick else (TRAJ_CHAINS, TRAJ_STEPS)

    src_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    rows = []
    summary = []
    for n in N_VALUES:
        ulp_n = np.spacing(float(n))
        tol = TOL_FACTOR * n * n * EPS
        worst_even, worst_odd3, worst_conserve = -np.inf, -np.inf, 0.0
        for seed in SEEDS:
            blocks = part_a(n, seed, m) + [part_b(n, seed, tc, ts)]
            for name, cnt, mi, pd, mr in blocks:
                for j in range(1, n + 1):
                    inc = float(mi[j - 1])
                    parity = "even" if j % 2 == 0 else "odd"
                    gated = (parity == "even")
                    ok = (inc <= tol) if gated else ""
                    rows.append([n, seed, name, j, parity, cnt,
                                 f"{inc:.17g}", f"{inc / ulp_n:.6g}",
                                 f"{tol:.6g}", ("PASS" if ok else "FAIL") if gated else "control",
                                 f"{float(pd[j - 1]):.6g}", f"{float(mr[j - 1]):.6g}",
                                 src_sha])
                    if parity == "even" and j < n:
                        worst_even = max(worst_even, inc)
                    if j == n:
                        worst_conserve = max(worst_conserve, abs(inc))
                    if j == 3 and n >= 4:
                        worst_odd3 = max(worst_odd3, inc)
        summary.append((n, worst_even, worst_even / ulp_n, tol,
                        worst_even <= tol, worst_odd3, worst_conserve / ulp_n))

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "cascade_monotone.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["N", "seed", "family", "j", "parity", "n_states",
                    "max_increase", "max_increase_ulps_of_N", "tol", "gate",
                    "p_strict_decrease", "mean_rel_decrease", "generator_sha16"])
        w.writerows(rows)

    print(f"wrote {path}  (generator sha16 {src_sha}, "
          f"{'QUICK' if args.quick else 'FULL'})")
    print(f"{'N':>4} {'max inc, even j<N':>20} {'ulps of N':>10} "
          f"{'tol':>10} {'gate':>5} {'odd j=3 max inc':>16} {'|E_N drift| ulps':>16}")
    all_pass = True
    for n, we, weu, tol, ok, wo, wc in summary:
        all_pass &= ok
        print(f"{n:>4} {we:>20.3e} {weu:>10.3g} {tol:>10.3e} "
              f"{'PASS' if ok else 'FAIL':>5} {wo:>16.3e} {wc:>16.3g}")
    print(f"VERDICT: {'PASS' if all_pass else 'FAIL'}  "
          f"(every even j < N at every N, both seeds, all families, "
          f"trajectories included)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
