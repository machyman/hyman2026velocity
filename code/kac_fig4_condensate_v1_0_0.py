#!/usr/bin/env python3
"""Generate Figure 4 (fig:condensate) of hyman2026velocity.

Panel (a)  Stationary density of a single velocity component under the sorted
           rule, against the random-pairing null control and the exact
           Maxwellian.  Semi-log y.  The sorted rule piles mass at the origin
           and thins the shoulder: the condensate.
Panel (b)  Survival function P(|v| > x), log-log, on the same three.  The
           sorted arm falls at once to a plateau at height about 2/N and then
           cuts off at |v| = sqrt(N).  Read together those two numbers say
           that two particles hold the entire energy and the rest are at
           rest.  The plateau is below the Maxwellian through the near tail,
           where the two-sigma mass is lost, and crosses far above it beyond
           x = 3, which is where the inflated fourth and sixth moments come
           from.  The moment inflation and the tail deficit are the same
           fact, not competing ones.

The figure exists to answer FTQ-3: the condensate is asserted throughout the
paper and had never been drawn.

Method.  The chain is the one used for Figure 2, `step_sphere` from
`kac_manuscript_data_v1_0_0.py`, reproduced here verbatim so this generator
stands alone.  Velocities live on the sphere sum v_i^2 = N; the rotation
conserves each pair's energy exactly, so no projection is applied and the
chain never leaves the sphere.  The null control is random pairing, which is
state independent and must reproduce the Maxwellian; it is the check that
licenses belief in the sorted arm.

Input   none (self-contained simulation)
Output  fig04_condensate.pdf, fig04_condensate.png, fig04_condensate.csv

Determinism.  Every run is seeded from SEEDS below and is reproducible on any
machine with the same numpy.  Matplotlib stamps a CreationDate into the PDF;
set SOURCE_DATE_EPOCH to pin it and make the output byte-reproducible:

    SOURCE_DATE_EPOCH=1788000000 python3 kac_fig4_condensate_v1_0_0.py

Without it the bytes differ in that field alone.

Usage:  python3 kac_fig4_condensate_v1_0_0.py [--outdir DIR] [--quick]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed style, held explicitly so the figure does not drift with rcParam
# defaults across matplotlib versions.  Matches Figure 3.
RED, BLUE, GREY = "#d62728", "#1f77b4", "grey"
FIGSIZE = (9.2, 3.5)
DPI_PNG = 150

# Simulation constants.  Two seeds, per the paper's two-seed discipline.
N_PART, N_CHAINS = 2048, 50
BURN, COLLECT, GAP = 2000, 40, 25
SEEDS = (813950001, 813950002)

# Exact Maxwellian references for the standard normal marginal.
M4_EXACT, M6_EXACT = 3.0, 15.0
TAIL2_EXACT = 0.045500263896358          # P(|v| > 2), two-sided


# One Kac step on the sphere.  The function below is reproduced BYTE-IDENTICALLY
# from kac_manuscript_data_v1_0_0.py, so this generator and the published data
# share one implementation of the dynamics.  Nothing is added to it, not even a
# docstring, so the claim is checkable by plain string comparison.  `sorted`
# pairs by ascending |v| (the biased rule); `random` pairs uniformly at random
# (the state-independent null control); `fixed` pairs adjacent indices.  This
# generator calls only the first two, but the branch is kept so the copy is
# complete.
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


def sample(rule, seed, burn, collect, gap):
    """Burn in, then pool decorrelated snapshots of every component."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N_CHAINS, N_PART))
    v *= np.sqrt(N_PART / np.sum(v * v, axis=1, keepdims=True))
    for _ in range(burn):
        step_sphere(v, rule, rng)
    pool = []
    for _ in range(collect):
        for _ in range(gap):
            step_sphere(v, rule, rng)
        pool.append(v.copy().ravel())
    return np.concatenate(pool)


def moments(s):
    """Fourth and sixth moments and the two-sigma tail mass."""
    s2 = s * s
    return float(np.mean(s2 * s2)), float(np.mean(s2 * s2 * s2)), \
        float(np.mean(np.abs(s) > 2.0))


def density(s, edges):
    """Normalized histogram on the given edges."""
    h, _ = np.histogram(s, bins=edges, density=True)
    return h


def survival(s, grid):
    """P(|v| > x) on the given grid, by sorted search."""
    a = np.sort(np.abs(s))
    return 1.0 - np.searchsorted(a, grid, side="right") / a.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--quick", action="store_true",
                    help="short run for shape checking; NOT the registered figure")
    a = ap.parse_args()
    burn, collect, gap = (200, 8, 10) if a.quick else (BURN, COLLECT, GAP)

    # Pool both seeds per arm; report the per-seed moments so the two-seed
    # agreement is visible rather than asserted.
    data, per_seed = {}, {}
    for rule in ("sorted", "random"):
        parts = []
        for sd in SEEDS:
            s = sample(rule, sd, burn, collect, gap)
            per_seed[(rule, sd)] = moments(s)
            parts.append(s)
        data[rule] = np.concatenate(parts)

    print(f"{'arm':10s} {'seed':>10s} {'m4':>12s} {'m6':>14s} {'P(|v|>2)':>10s}")
    for (rule, sd), (m4, m6, t2) in per_seed.items():
        print(f"{rule:10s} {sd:>10d} {m4:>12.4g} {m6:>14.6g} {t2:>10.5f}")
    print(f"{'Maxwellian':10s} {'exact':>10s} {M4_EXACT:>12.4g} "
          f"{M6_EXACT:>14.6g} {TAIL2_EXACT:>10.5f}")

    # ---- panel data -----------------------------------------------------
    edges = np.linspace(-4.0, 4.0, 161)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    d_sorted, d_random = density(data["sorted"], edges), density(data["random"], edges)
    d_exact = np.exp(-0.5 * ctr ** 2) / np.sqrt(2.0 * np.pi)

    grid = np.logspace(np.log10(0.05), np.log10(60.0), 220)
    s_sorted, s_random = survival(data["sorted"], grid), survival(data["random"], grid)
    from math import erfc, sqrt
    s_exact = np.array([erfc(x / sqrt(2.0)) for x in grid])
    # An empirical survival function that has run out of samples reads exactly
    # zero.  Plotting that on a log axis draws a cliff that is an artefact of
    # the sample size, so mask it rather than draw it.
    m_sorted = np.where(s_sorted > 0, s_sorted, np.nan)
    m_random = np.where(s_random > 0, s_random, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    ax1.semilogy(ctr, d_exact, "-", color=GREY, lw=2.2, label="Maxwellian, exact")
    ax1.semilogy(ctr, d_random, "--", color=BLUE, lw=1.6,
                 label="random pairing, null control")
    ax1.semilogy(ctr, d_sorted, "-", color=RED, lw=1.8, label="sorted rule")
    ax1.set_xlabel(r"$v$")
    ax1.set_ylabel("stationary density")
    ax1.set_xlim(-4.0, 4.0)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax1.set_title("(a) the condensate", fontsize=10)

    ax2.loglog(grid, s_exact, "-", color=GREY, lw=2.2, label="Maxwellian, exact")
    ax2.loglog(grid, m_random, "--", color=BLUE, lw=1.6,
               label="random pairing, null control")
    ax2.loglog(grid, m_sorted, "-", color=RED, lw=1.8, label="sorted rule")
    ax2.axhline(2.0 / N_PART, color=RED, lw=0.8, ls=":", alpha=0.7)
    ax2.axvline(np.sqrt(N_PART), color=RED, lw=0.8, ls=":", alpha=0.7)
    ax2.text(0.075, 2.0 / N_PART * 1.5, "$2/N$", color=RED, fontsize=8)
    ax2.text(np.sqrt(N_PART) * 0.30, 2.2e-7, r"$\sqrt{N}$", color=RED, fontsize=8)
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$P(|v| > x)$")
    ax2.set_xlim(0.05, 60.0)
    ax2.set_ylim(1e-7, 1.5)
    ax2.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax2.set_title("(b) two particles hold the energy", fontsize=10)

    ax1.set_ylim(1e-4, 3e1)
    for ax in (ax1, ax2):
        ax.grid(True, which="major", alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.outdir, f"fig04_condensate.{ext}"),
                    dpi=DPI_PNG if ext == "png" else None)
    plt.close(fig)

    # ---- deposit the plotted numbers ------------------------------------
    with open(os.path.join(a.outdir, "fig04_condensate.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "x", "sorted", "random", "maxwellian"])
        for x, p, q, r in zip(ctr, d_sorted, d_random, d_exact):
            w.writerow(["a_density", f"{x:.6f}", f"{p:.10g}", f"{q:.10g}", f"{r:.10g}"])
        for x, p, q, r in zip(grid, s_sorted, s_random, s_exact):
            w.writerow(["b_survival", f"{x:.6f}", f"{p:.10g}", f"{q:.10g}", f"{r:.10g}"])
    return per_seed


if __name__ == "__main__":
    main()
