#!/usr/bin/env python3
"""Generate Figure 2 (fig:finiteN) of hyman2026obstruction.

Kolmogorov-Smirnov distance between the empirical law of a single coordinate
after many collision steps and the Beta(1/2,(N-1)/2) marginal of sigma, against
particle number, under three pairing rules.

Input   manuscript_ks_sweep.csv  (written by kac_manuscript_data_v1_0_0.py)
Output  fig02_finite_N.pdf, fig02_finite_N.png

Why this exists. The registry named kac_manuscript_data_v1_0_0.py as this
figure's generator. That script writes the sweep CSV but never the PDF, so the
figure was recorded as reproducible while no deposited script drew it. The same
gap was closed for Figure 3 earlier; a clean-room harness found this one.

Determinism. Matplotlib stamps a CreationDate into the PDF. Pin it with
    SOURCE_DATE_EPOCH=1786822750 python3 kac_fig2_finiteN_v1_0_0.py

Usage:  python3 kac_fig2_finiteN_v1_0_0.py [--indir DIR] [--outdir DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "manuscript_ks_sweep.csv"
RED, GREEN, BLUE, GREY = "#d62728", "#2ca02c", "#1f77b4", "grey"
FIGSIZE = (5.2, 3.6)
DPI_PNG = 150

SERIES = [
    ("ks_sorted", "sorted (state-dependent)", RED, "o-"),
    ("ks_fixed", "fixed matching (state-independent)", GREEN, "s-"),
    ("ks_random", "random matching (null control)", BLUE, "^-"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=".")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    d = np.genfromtxt(os.path.join(a.indir, CSV), delimiter=",", names=True)
    order = np.argsort(d["N"])
    N = np.asarray(d["N"])[order]
    x = np.arange(len(N))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, label, colour, style in SERIES:
        ax.semilogy(x, np.asarray(d[key])[order], style, color=colour,
                    lw=1.5, ms=5, label=label)

    # the N = 2 exceptional case, where only one matching exists
    ax.annotate("$N=2$: one matching,\nno state dependence possible",
                xy=(x[0], d["ks_sorted"][order][0]),
                xytext=(1.15, 1.6e-2), fontsize=7.5, color=GREY,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n)}" for n in N])
    ax.set_xlabel("particle number $N$", fontsize=10)
    ax.set_ylabel(r"KS distance from Beta$(1/2,\,(N-1)/2)$", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()

    pdf = os.path.join(a.outdir, "fig02_finite_N.pdf")
    png = os.path.join(a.outdir, "fig02_finite_N.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI_PNG)
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    main()
