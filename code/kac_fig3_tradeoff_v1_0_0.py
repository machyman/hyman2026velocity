#!/usr/bin/env python3
"""Generate Figure 3 (fig:tradeoff) of hyman2026obstruction.

Panel (a)  Mean squared error against block width w, aligned angles, sixty
           randomizations.  Monotone on both observables: no interior optimum.
Panel (b)  The cross-term ratio C/Cbar, computable in O(N) from a single state
           before any trajectory is run, against the bias subsequently measured.

Input   phasef_part2_rounded.csv   (registered; produced by kac_phasef_v2_1_0.py)
Output  fig03_tradeoff.pdf, fig03_tradeoff.png

This script is a reconstruction.  The figure was originally drawn by an inline
script that was never written to disk; the data behind it was always deposited
and hashed.  The reconstruction is byte-verified against the registered PDF.

Determinism.  Matplotlib stamps a CreationDate into the PDF.  Set
SOURCE_DATE_EPOCH to pin it and make the output byte-reproducible:

    SOURCE_DATE_EPOCH=1786822750 python3 kac_fig3_tradeoff_v1_0_0.py

which is the timestamp carried by the registered fig03_tradeoff.pdf
(2026-08-15 19:39:10 UTC).  Without it the bytes differ in that field alone.

Usage:  python3 kac_fig3_tradeoff_v1_0_0.py [--indir DIR] [--outdir DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "phasef_part2_rounded.csv"

# Fixed style.  Held explicitly so the figure does not drift with rcParam
# defaults across matplotlib versions.
RED, BLUE, GREEN, PURPLE, GREY = "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "grey"
FIGSIZE = (9.2, 3.5)
DPI_PNG = 150


def load(indir):
    """Read the registered input, one row per block width."""
    d = np.genfromtxt(os.path.join(indir, CSV), delimiter=",", names=True)
    order = np.argsort(d["w"])
    return {k: np.asarray(d[k])[order] for k in d.dtype.names}


def panel_a(ax, d):
    """MSE and variance against block width, on evenly spaced categorical x."""
    x = np.arange(len(d["w"]))
    ax.semilogy(x, d["mse_v4"], "o-", color=RED, lw=1.6, ms=5,
                label=r"MSE $\langle v^4 \rangle$")
    ax.semilogy(x, d["mse_absv"], "s-", color=BLUE, lw=1.6, ms=5,
                label=r"MSE $\langle |v| \rangle$")
    ax.semilogy(x, d["var_absv"], "^--", color=GREEN, lw=1.6, ms=5, alpha=0.8,
                label=r"variance $\langle |v| \rangle$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(w)}" for w in d["w"]])
    ax.set_xlabel(r"block width $w$   (2 = sorted, $N$ = random)")
    ax.set_ylabel("error")
    ax.set_title("(a) MSE is monotone; no interior optimum", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def panel_b(ax, d):
    """Measured bias against the a priori cross-term ratio."""
    ratio = d["c_over_cbar"]
    bias = np.abs(d["bias_absv"])
    ax.plot(ratio, bias, "o-", color=PURPLE, lw=1.6, ms=7)
    for r, b, w in zip(ratio, bias, d["w"]):
        ax.annotate(f"$w$={int(w)}", (r, b), textcoords="offset points",
                    xytext=(7, -9), fontsize=7, color=GREY)
    ax.axvline(1.0, color=GREY, ls=":", lw=1.2)
    ax.annotate("exchangeable value", xy=(1.02, 0.34), xytext=(1.30, 0.30),
                fontsize=8, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.set_xlabel(r"$C/\overline{C}$   (computable before any trajectory)")
    ax.set_ylabel(r"measured $|$bias $\langle |v| \rangle|$")
    ax.set_title(r"(b) $C/\overline{C}$ predicts the bias a priori", fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default=".")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    d = load(a.indir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    panel_a(ax1, d)
    panel_b(ax2, d)
    fig.tight_layout()

    pdf = os.path.join(a.outdir, "fig03_tradeoff.pdf")
    png = os.path.join(a.outdir, "fig03_tradeoff.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI_PNG)
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    main()
