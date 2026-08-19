#!/usr/bin/env python3
"""
kac_remaining_v1_0_0.py -- DISCLOSED DEVELOPMENT WORK. Zero offsets.

The last three experiments whose numbers exist only as printed output:

  A  moment ladder across the block width        -> Section 6, experiment 2
  B  transfer test, the Hilbert sort on the       -> Section 7
     campaign's own registered t = 0 loads
  C  spatial-cell check against both nulls        -> Section 8

Run through generator_template_v1_0_0, so every quantity carries a fragility
class and a standard error, the whole thing runs under two seeds, and only
digits agreeing between them are emitted.  This is the template's first use
on real work, and is as much a test of the template as of the physics.

A note on classes.  C/Cbar is reported as a MEAN over independent draws of a
per-draw ratio, not as a ratio of two aggregates, so it is Class B and its
standard error is the ordinary sd/sqrt(n).  That standard error is the
resolution of the diagnostic, which Sections 5 and 8 both need and neither
previously had.
"""

import csv
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "/home/claude/kac")
from generator_template_v1_0_0 import Q, Report          # noqa: E402

N_PART, N_STEPS = 2048, 800
TAIL_GAUSS = 0.045500263896358
WIDTHS = (2, 32, 512, N_PART)


# ------------------------------------------------------------------ shared
def cbar_global(v):
    n = v.size
    return (n * n - float(np.sum(v ** 4))) / (2.0 * (n - 1))


def block_pairs(v, w, rng):
    n = v.size
    o = np.argsort(np.abs(v), kind="stable")
    if w >= n:
        o = rng.permutation(n)
        return o[0::2], o[1::2]
    nb = n // w
    head = o[: nb * w].reshape(nb, w)
    head = np.take_along_axis(head, np.argsort(rng.random(head.shape), axis=1), axis=1)
    o = np.concatenate([head.ravel(), o[nb * w:]])
    return o[0::2], o[1::2]


def evolve(w, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=N_PART)
    v -= v.mean(); v *= np.sqrt(N_PART / np.dot(v, v))
    for _ in range(N_STEPS):
        ia, ib = block_pairs(v, w, rng)
        t = 2.0 * np.pi * rng.random(ia.size)
        r = np.hypot(v[ia], v[ib])
        v[ia] = r * np.cos(t); v[ib] = r * np.sin(t)
        v -= v.mean(); v *= np.sqrt(N_PART / np.dot(v, v))
    return (float(np.mean(v ** 2)),
            float(np.mean(v ** 4)) / 3.0,
            float(np.mean(v ** 6)) / 15.0,
            float(np.mean(np.abs(v) > 2.0)) / TAIL_GAUSS)


def c_ratio_block(w, seed, draws=12):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        v = rng.normal(size=N_PART)
        v -= v.mean(); v *= np.sqrt(N_PART / np.dot(v, v))
        ia, ib = block_pairs(v, w, rng)
        out.append(float(np.sum(v[ia] ** 2 * v[ib] ** 2)) / cbar_global(v))
    return np.array(out)


# --------------------------------------------------------------- B and C
ARCHIVE = "discrepancy_campaign_R4_FULL_results_2026-08-12.zip"

def _ensure(npart):
    """Standalone: extract the registered snapshot from its archive if absent."""
    import os, zipfile
    f = f"snapshot_S-C4b_Np{npart}_o74_t0.npz"
    if not os.path.exists(f):
        if not os.path.exists(ARCHIVE):
            raise SystemExit(f"need {f} or the registered archive {ARCHIVE}")
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extract(f)
    return f

def load(npart):
    d = np.load(_ensure(npart))
    x, v = d["x"], d["v"].astype(float)
    return x, v * np.sqrt(npart / np.sum(v * v))


def hilbert_ratio(npart, order=10):
    from hilbertcurve.hilbertcurve import HilbertCurve
    x, v = load(npart)
    hc = HilbertCurve(order, 2); side = 2 ** order - 1
    xi = np.clip((x * side).astype(int), 0, side)
    vi = np.clip((norm.cdf(v / v.std()) * side).astype(int), 0, side)
    o = np.argsort(np.array(hc.distances_from_points(np.stack([xi, vi], 1))), kind="stable")
    ia, ib = o[0::2], o[1::2]
    return float(np.sum(v[ia] ** 2 * v[ib] ** 2)) / cbar_global(v)


def cell_ratios(npart, ncell, seed, draws=12):
    x, v = load(npart)
    rng = np.random.default_rng(seed)
    cell = np.clip((x * ncell).astype(int), 0, ncell - 1)
    g, wc = [], []
    for _ in range(draws):
        A, B, cb = [], [], 0.0
        left = []
        for c in range(ncell):
            m = np.flatnonzero(cell == c)
            if m.size < 2:
                left += m.tolist(); continue
            m = rng.permutation(m); k = m.size // 2 * 2
            A.append(m[:k:2]); B.append(m[1:k:2])
            if m.size % 2:
                left.append(int(m[-1]))
            s2 = float(np.sum(v[m] ** 2)); q4 = float(np.sum(v[m] ** 4))
            cb += (s2 * s2 - q4) / (2 * (m.size - 1))
        if len(left) >= 2:
            lo = rng.permutation(np.array(left)); k = lo.size // 2 * 2
            A.append(lo[:k:2]); B.append(lo[1:k:2])
        ia, ib = np.concatenate(A), np.concatenate(B)
        C = float(np.sum(v[ia] ** 2 * v[ib] ** 2))
        g.append(C / cbar_global(v)); wc.append(C / cb)
    return np.array(g), np.array(wc)


# ------------------------------------------------------------------ report
def experiment(seed):
    out = {}
    for w in WIDTHS:
        reps = np.array([evolve(w, seed * 1000 + w * 10 + k) for k in range(8)])
        for j, nm in enumerate(("m2", "m4", "m6", "tail")):
            out[f"A_w{w}_{nm}"] = Q(reps[:, j], "mean", note=f"block width {w}")
        out[f"A_w{w}_CoverCbar"] = Q(c_ratio_block(w, seed * 7 + w), "mean",
                                     note="per-draw ratio, averaged")
    for npart in (2048, 8192):
        out[f"B_Np{npart}_hilbert"] = Q([hilbert_ratio(npart)], "exact",
                                        note="deterministic on a single registered load")
        gl, wc = cell_ratios(npart, 64, seed * 13 + npart)
        out[f"C_Np{npart}_cell_global"] = Q(gl, "mean", note="global null")
        out[f"C_Np{npart}_cell_within"] = Q(wc, "mean", note="within-cell null")
    return out


if __name__ == "__main__":
    rep = Report("remaining", experiment, seeds=(21, 22),
                 source="/home/claude/kac/kac_remaining_v1_0_0.py").run()
    for r in rep.rows:
        print(f"  {r['quantity']:>24} {r['value']:>12}  +/- {r['se']:>9}  "
              f"class {r['fragility_class']}  digits {r['digits_shown']}/{r['digits_from_se']}")
    rep.write("remaining_results.csv")
