#!/usr/bin/env python3
"""
kac_general_law_v1_0_0.py  --  W11 audit, Session 41.

QUESTION.  Master Plan 29.3 asserts that Theorem 9.2 splits: the condensation
half generalises to any scattering law with a density bounded below on an
interval longer than a quarter turn, while the limiting phase and the moment
law stay specific to the uniform law, because "under a general law the top
pair's phase converges to that law's stationary distribution on the circle,
and the moment formula changes with it."

That assertion is tested here, not assumed.  The Fourier argument says it is
too conservative: the phase advances by convolution, phi_{n+1} = phi_n + Theta,
and on the circle nu^{*n} converges to Haar measure unless nu sits on a coset
of a proper closed subgroup.  The proper closed subgroups of the circle are
finite, and a law with a density bounded below on an interval has uncountable
support, so it cannot be one.  Every non-zero Fourier coefficient then has
modulus strictly below one.  Since cos^{2k} is a trigonometric polynomial of
degree 2k, E[cos^{2k} Phi_n] is a finite combination of those coefficients and
converges to its value under Haar.  The moment law should therefore SURVIVE.

THE COMPLICATION, and the reason this is measured rather than argued.  The
phase walk is autonomous only while the top pair keeps its identity.  v1_55_1
recorded that a lower particle can overtake with probability of order
sqrt(E_{N-2}/N) per step and that the sum of those may diverge, so the top
pair may change infinitely often and the walk is reset each time it does.
Under the uniform law this is harmless: the post-collision phase is exactly
uniform at every step whatever the pair's identity.  Under a general law
convergence needs a run of steps on one pair, so the measurement is the honest
arbiter.

PREDICTIONS, recorded before running:
  L1 uniform            -- condensation yes; moments match (baseline).
  L2 arc, eps = 0.30    -- condensation yes; moments match if the argument holds.
  L3 arc, eps = 0.05    -- as L2, tight hypothesis; slower phase mixing.
  L4 smooth non-uniform -- as L2.
  L5 irrational rotation-- aperiodic but NO density, so outside the theorem's
                           hypothesis.  Probe only: does it condense anyway?
  L6 quarter-turn atom  -- PERIODIC, phase confined to a 4-point lattice.
                           CONTROL: the moment law must FAIL here.  If it does
                           not, the measurement is not sensitive and nothing
                           above may be believed.

GATE.  L1 must reproduce m4 = 3N/4 and m6 = 5N^2/8 to within the two-seed
spread, and L6 must visibly depart from them.  Both must hold or the run is
void.

Outputs general_law.csv.  Two seeds, byte-reproducible.  No PNG (F-21).
"""

import numpy as np
import csv
import sys

BASE = 813950001


# ---------------------------------------------------------------------------
# step_sphere, reproduced VERBATIM from kac_manuscript_data_v1_0_0.py so that
# the probe and the published figures share one implementation of the dynamics.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# General-law step: ROTATE the pair by Theta rather than RESAMPLE its phase.
# With Theta uniform this is the same kernel as step_sphere, which test 1 checks.
# ---------------------------------------------------------------------------
def step_general(v, draw_theta, rng):
    n_chains, n_part = v.shape
    o = np.argsort(np.abs(v), axis=1, kind="stable")
    ia, ib = o[:, 0::2], o[:, 1::2]
    va = np.take_along_axis(v, ia, axis=1)
    vb = np.take_along_axis(v, ib, axis=1)
    r = np.hypot(va, vb)
    phi = np.arctan2(vb, va)
    t = phi + draw_theta(r.shape, rng)
    np.put_along_axis(v, ia, r * np.cos(t), axis=1)
    np.put_along_axis(v, ib, r * np.sin(t), axis=1)
    return v


# ---------------------------------------------------------------------------
# Scattering laws.  Each returns Theta of the requested shape.
# ---------------------------------------------------------------------------
GOLDEN = 2.0 * np.pi * ((1.0 + 5.0 ** 0.5) / 2.0 % 1.0)


def law_uniform(shape, rng):
    return 2.0 * np.pi * rng.random(shape)


def _arc(eps):
    width = np.pi / 2.0 + eps

    def f(shape, rng):
        return width * rng.random(shape)
    return f


def law_smooth(shape, rng):
    # Density bounded below on an interval of length pi/2 + 0.4: a raised
    # cosine on that interval, floored so the lower bound c > 0 is explicit.
    width = np.pi / 2.0 + 0.4
    x = rng.random(shape)
    # inverse-CDF-free rejection-lite: mix a uniform floor with a cosine bump
    u = rng.random(shape)
    bump = width * (0.5 + 0.5 * np.sin(np.pi * (x - 0.5)))
    return np.where(u < 0.35, width * x, bump)


def law_irrational(shape, rng):
    return np.full(shape, GOLDEN)


def law_quarter(shape, rng):
    return np.full(shape, np.pi / 2.0)


LAWS = [
    ("L1_uniform",     law_uniform,      "uniform on the circle"),
    ("L2_arc_eps0.30", _arc(0.30),       "uniform on an arc of length pi/2+0.30"),
    ("L3_arc_eps0.05", _arc(0.05),       "uniform on an arc of length pi/2+0.05"),
    ("L4_smooth",      law_smooth,       "non-uniform density, floored, width pi/2+0.40"),
    ("L5_irrational",  law_irrational,   "atom at an irrational rotation (no density)"),
    ("L6_quarter",     law_quarter,      "atom at pi/2 (PERIODIC control)"),
]


def energy_below_top(v):
    """E_{N-2}: energy held by all but the two fastest, per chain."""
    s = np.sort(v * v, axis=1)
    return s[:, :-2].sum(axis=1)


def run(name, draw, n_part, seed, chains, burn, collect, gap):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(chains, n_part))
    v *= np.sqrt(n_part / (v * v).sum(axis=1, keepdims=True))

    for _ in range(burn):
        v = step_general(v, draw, rng)

    m4 = m6 = tail = e_top = maxv = 0.0
    for _ in range(collect):
        for _ in range(gap):
            v = step_general(v, draw, rng)
        m4 += (v ** 4).mean()
        m6 += (v ** 6).mean()
        tail += (np.abs(v) > 2.0).mean()
        e_top += energy_below_top(v).mean()
        maxv += (np.abs(v).max(axis=1) / np.sqrt(n_part)).mean()
    k = float(collect)
    return dict(m4=m4 / k, m6=m6 / k, tail=tail / k,
                e_below_top=e_top / k, maxv_ratio=maxv / k)


def main(quick=False):
    Ns = [16, 64] if quick else [16, 64, 256]
    chains = 60 if quick else 200
    burn = 800 if quick else 4000
    collect = 12 if quick else 40
    gap = 10 if quick else 25

    rows = []
    print(f"{'law':<16}{'N':>5}{'seed':>4}"
          f"{'m4':>12}{'3N/4':>10}{'m6':>14}{'5N^2/8':>12}"
          f"{'E_belowtop':>12}{'max|v|/sqN':>12}")
    print("-" * 110)
    for name, draw, desc in LAWS:
        for n_part in Ns:
            pred4 = 3.0 * n_part / 4.0
            pred6 = 5.0 * n_part ** 2 / 8.0
            for si, seed in enumerate((BASE, BASE + 1)):
                r = run(name, draw, n_part, seed + 7919 * n_part,
                        chains, burn, collect, gap)
                d4 = 100.0 * (r["m4"] - pred4) / pred4
                d6 = 100.0 * (r["m6"] - pred6) / pred6
                print(f"{name:<16}{n_part:>5}{si:>4}"
                      f"{r['m4']:>12.4g}{pred4:>10.4g}"
                      f"{r['m6']:>14.5g}{pred6:>12.5g}"
                      f"{r['e_below_top']:>12.4g}{r['maxv_ratio']:>12.6f}")
                rows.append(dict(law=name, description=desc, N=n_part,
                                 seed_index=si, m4=r["m4"], m4_pred=pred4,
                                 m4_pct=d4, m6=r["m6"], m6_pred=pred6,
                                 m6_pct=d6, tail=r["tail"],
                                 e_below_top=r["e_below_top"],
                                 maxv_ratio=r["maxv_ratio"]))
        print("-" * 110)
        # Incremental flush: a killed run keeps every law it finished.
        with open("general_law.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- GATE -------------------------------------------------------------
    def pct(law, field):
        return [abs(r[field]) for r in rows if r["law"] == law]

    l1_ok = max(pct("L1_uniform", "m4_pct")) < 5.0 and \
            max(pct("L1_uniform", "m6_pct")) < 10.0
    l6_bad = max(pct("L6_quarter", "m4_pct")) > 10.0
    print()
    print(f"GATE  L1 uniform reproduces the moment law      : "
          f"{'PASS' if l1_ok else 'FAIL'}  "
          f"(max |m4 dev| {max(pct('L1_uniform','m4_pct')):.2f}%)")
    print(f"GATE  L6 periodic control departs from it       : "
          f"{'PASS' if l6_bad else 'FAIL'}  "
          f"(max |m4 dev| {max(pct('L6_quarter','m4_pct')):.2f}%)")
    print(f"VERDICT: {'PASS' if (l1_ok and l6_bad) else 'VOID'}")
    return 0 if (l1_ok and l6_bad) else 1


if __name__ == "__main__":
    sys.exit(main(quick="--quick" in sys.argv))
