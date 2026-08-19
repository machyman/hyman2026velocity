#!/usr/bin/env python3
"""Generate Figure 1 (fig:spectrum) of hyman2026obstruction.

The mechanism figure.  For one state of N = 6 particles it shows

  (a) the six speeds in rank order, with the sorted pairing drawn above the
      axis (adjacent ranks) and the extremal pairing below (rank i with rank
      N+1-i), so the two extreme rules are visible as pictures rather than
      definitions;

  (b) every one of the 15 perfect matchings placed on the C axis, with the
      sorted matching at the right end and the extremal at the left, and the
      exchangeable value Cbar marked.  Cbar is the mean of the 15 points, which
      is Lemma 4.4 seen directly.  The top axis carries the resulting one-step
      fourth moment (3/4)(Q + 2C) from Lemma 4.2, so the same picture shows
      that the pairing enters only through C and that moving C moves the
      fourth moment.

Input   none; the state is generated from the seed recorded below.
Output  fig01_spectrum.pdf, fig01_spectrum.png, spectrum_N6.csv

Determinism.  Matplotlib stamps a CreationDate into the PDF.  Pin it with
    SOURCE_DATE_EPOCH=1786822750 python3 kac_fig1_spectrum_v1_0_0.py
Usage:  python3 kac_fig1_spectrum_v1_0_0.py [--outdir DIR]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 813960000
N = 6
RED, BLUE, GREY, PURPLE = "#d62728", "#1f77b4", "grey", "#9467bd"
FIGSIZE = (9.2, 3.2)
DPI_PNG = 150


def matchings(idx):
    """All perfect matchings of a list of even length."""
    if not idx:
        return [[]]
    out, a = [], idx[0]
    for j in range(1, len(idx)):
        b = idx[j]
        rest = [k for k in idx if k not in (a, b)]
        out += [[(a, b)] + sub for sub in matchings(rest)]
    return out


def state(seed, n):
    """One draw from sigma: uniform on {sum v_i^2 = n}, returned rank-ordered."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)
    v *= np.sqrt(n / np.dot(v, v))
    return np.sort(np.abs(v))


def C_of(x, P):
    return sum(x[a] * x[b] for a, b in P)


def panel_a(ax, speeds):
    """The two extreme rules drawn as arcs over the rank-ordered speeds."""
    pos = np.arange(1, N + 1)
    ax.plot(pos, np.zeros(N), "o", color="black", ms=6, zorder=3)
    for p, s in zip(pos, speeds):
        ax.annotate(f"{s:.2f}", (p, 0), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=7, color=GREY)
    sorted_pairs = [(1, 2), (3, 4), (5, 6)]
    extremal_pairs = [(1, 6), (2, 5), (3, 4)]
    for a, b in sorted_pairs:
        ax.annotate("", xy=(b, 0), xytext=(a, 0),
                    arrowprops=dict(arrowstyle="-", color=RED, lw=1.6,
                                    connectionstyle="arc3,rad=-0.55"))
    for a, b in extremal_pairs:
        ax.annotate("", xy=(b, 0), xytext=(a, 0),
                    arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.6,
                                    connectionstyle="arc3,rad=0.42"))
    ax.text(3.5, 0.62, "sorted: pair adjacent ranks", color=RED,
            fontsize=8, ha="center")
    ax.text(3.5, -0.70, "extremal: pair rank $i$ with rank $N{+}1{-}i$",
            color=BLUE, fontsize=8, ha="center")
    ax.set_xlim(0.3, 6.7)
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([])
    ax.set_xticks(pos)
    ax.set_xlabel("rank by speed $|v_i|$", fontsize=9)
    ax.set_title("(a) the two extreme rules on one state, $N=6$", fontsize=9, pad=26)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def panel_b(ax, x, Ms, Cvals, Cbar, Q):
    """All 15 matchings on the C axis, with the one-step fourth moment on top."""
    y = np.zeros(len(Cvals))
    ax.plot(Cvals, y, "o", color=PURPLE, ms=6, alpha=0.55, zorder=2,
            label="the 15 matchings")
    i_max, i_min = int(np.argmax(Cvals)), int(np.argmin(Cvals))
    ax.plot([Cvals[i_max]], [0], "o", color=RED, ms=9, zorder=4)
    ax.plot([Cvals[i_min]], [0], "o", color=BLUE, ms=9, zorder=4)
    ax.axvline(Cbar, color=GREY, ls=":", lw=1.3)
    ax.annotate("sorted\n(maximum)", (Cvals[i_max], 0), textcoords="offset points",
                xytext=(-4, 24), ha="center", fontsize=8, color=RED)
    ax.annotate("extremal\n(minimum)", (Cvals[i_min], 0), textcoords="offset points",
                xytext=(4, 24), ha="center", fontsize=8, color=BLUE)
    ax.annotate(r"$\overline{C}$, the exchangeable value" "\n" "and the mean of the points",
                (Cbar, 0), textcoords="offset points", xytext=(9, -46),
                ha="left", fontsize=8, color=GREY)
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([])
    ax.set_xlabel(r"$C(P,v) = \sum_{(a,b) \in P} v_a^2 v_b^2$", fontsize=9)
    ax.set_title(r"(b) the pairing enters only through $C$", fontsize=9, pad=26)
    for side in ("left", "right"):
        ax.spines[side].set_visible(False)
    sec = ax.secondary_xaxis(
        "top", functions=(lambda c: 0.75 * (Q + 2 * c),
                          lambda q: (q / 0.75 - Q) / 2))
    sec.set_xlabel(r"one-step fourth moment $\mathbb{E}_{\theta}[Q(Tv)\mid v]"
                   r" = \frac{3}{4}(Q + 2C)$", fontsize=8)
    sec.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    speeds = state(SEED, N)
    x = speeds ** 2
    Q = float(np.sum(x ** 2))
    Ms = matchings(list(range(N)))
    Cvals = np.array([C_of(x, P) for P in Ms])
    Cbar = (N * N - Q) / (2 * (N - 1))

    # Lemma 4.4 says Cbar is the average over all matchings; check it here.
    assert abs(Cbar - Cvals.mean()) < 1e-12, "Cbar is not the mean of the matchings"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE,
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    panel_a(ax1, speeds)
    panel_b(ax2, x, Ms, Cvals, Cbar, Q)
    fig.tight_layout()

    pdf = os.path.join(a.outdir, "fig01_spectrum.pdf")
    png = os.path.join(a.outdir, "fig01_spectrum.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI_PNG)
    plt.close(fig)

    with open(os.path.join(a.outdir, "spectrum_N6.csv"), "w", newline="\n") as fh:
        w = csv.writer(fh)
        w.writerow(["matching", "C", "one_step_EQ"])
        for P, c in zip(Ms, Cvals):
            w.writerow(["|".join(f"({i+1},{j+1})" for i, j in P),
                        f"{c:.12g}", f"{0.75*(Q+2*c):.12g}"])
        w.writerow(["Cbar", f"{Cbar:.12g}", f"{0.75*(Q+2*Cbar):.12g}"])
        w.writerow(["Q", f"{Q:.12g}", ""])
    print(f"wrote {pdf}\nwrote {png}\nCbar = {Cbar:.6f}  mean = {Cvals.mean():.6f}  "
          f"min = {Cvals.min():.6f}  max = {Cvals.max():.6f}")


if __name__ == "__main__":
    main()
