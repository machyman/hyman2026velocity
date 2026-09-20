#!/usr/bin/env python3
"""
kac_n4_stationary_probe_v1_0_0.py -- EXPLORATORY. NOT a registered generator.

Nothing here is cited by the manuscript. This script exists so that the
Session 38 probe survives a branch; promote it to a registered generator only
if G8 is reopened and the result enters the paper.

WHAT IT ESTABLISHES (Kac model on the sphere sum v_i^2 = N, sorted pairing)

  1. REDUCTION.  At N = 4 the chain is one-dimensional.  Let E1 be the energy
     of the two smallest particles.  The within-pair rotation preserves each
     pair's energy, so the only quantity carried across a step is the split;
     the sort then re-partitions it.  Verified empirically: two structurally
     different families of states at the same E1 give the same one-step law
     of E1', in mean and across quantiles, at E1 ~ 0.15 and E1 ~ 0.35.

  2. THE LIMIT.  E1 -> 0.  Two particles come to rest; the other two carry the
     whole energy on a circle of radius sqrt(N) = 2 with uniform phase.  So the
     limit law is that of (0, 0, 2 cos T, 2 sin T) up to permutation, T uniform.
     Its moments are m_{2k} = 2^(k-1) (2k-1)!! / k!,  giving m2 = 1, m4 = 3,
     m6 = 10, m8 = 35.
     The correct equilibrium (uniform on the sphere) has
     m_{2k} = N^k (2k-1)!! / (N (N+2) ... (N+2k-2)), giving m4 = 2, m6 = 5.
     At N = 4 the sorted rule inflates the 2k-th moment by exactly (k+1)/2.

  MEASURED (2000 steps, 40000 chains, two seeds):
     seed 11  m4 = 3.00842   m6 = 10.05054
     seed 12  m4 = 3.00008   m6 = 10.00050
     uniform  m4 =  2.00649  m6 =  5.03226   (exact 2 and 5)

WHAT IT DOES NOT ESTABLISH
  Convergence is not proved.  The decay of E1 is monotone in simulation but
  no supermartingale argument is given.  The limit is IDENTIFIED BY MOMENTS,
  not proved.  Section 9's conjecture stands until that gap is closed.

Usage:  python3 kac_n4_stationary_probe_v1_0_0.py [--part reduce|limit|both]
"""
import argparse
import numpy as np

N = 4


def step(v, rng):
    """One Kac collision step under the sorted rule: pair adjacent |v| ranks."""
    o = np.argsort(np.abs(v), axis=1, kind="stable")
    ia, ib = o[:, 0::2], o[:, 1::2]
    va = np.take_along_axis(v, ia, axis=1)
    vb = np.take_along_axis(v, ib, axis=1)
    r = np.hypot(va, vb)
    t = 2.0 * np.pi * rng.random(r.shape)
    np.put_along_axis(v, ia, r * np.cos(t), axis=1)
    np.put_along_axis(v, ib, r * np.sin(t), axis=1)
    return v


def energy_low(v):
    """E1: energy carried by the two slowest particles."""
    s = np.sort(v * v, axis=1)
    return s[:, 0] + s[:, 1]


def init(m, rng):
    v = rng.normal(size=(m, N))
    return v * np.sqrt(N / np.sum(v * v, axis=1, keepdims=True))


def reduction(rng):
    print("reduction: law of E1' given the state, at matched E1")
    for target in (0.15, 0.35):
        a = init(300000, rng)
        a = a[np.abs(energy_low(a) - target) < 0.004][:8000].copy()
        b = step(init(300000, rng), rng)
        b = b[np.abs(energy_low(b) - target) < 0.004][:8000].copy()
        for name, x in (("A", a), ("B", b)):
            y = energy_low(step(x.copy(), rng))
            print(f"  E1~{target}  {name}: n={len(x)} mean={y.mean():.5f} "
                  f"q10={np.quantile(y,.1):.4f} q50={np.quantile(y,.5):.4f} "
                  f"q90={np.quantile(y,.9):.4f}")


def limit():
    print("limit: prediction m2=1 m4=3 m6=10;  uniform-on-sphere m4=2 m6=5")
    for seed in (11, 12):
        rng = np.random.default_rng(seed)
        v = init(40000, rng)
        for _ in range(2000):
            step(v, rng)
        print(f"  seed {seed}: E1={energy_low(v).mean():.3e}  m2={np.mean(v**2):.6f} "
              f"m4={np.mean(v**4):.5f}  m6={np.mean(v**6):.5f}")
    u = init(40000, np.random.default_rng(99))
    print(f"  uniform  : E1={energy_low(u).mean():.6f}  m2={np.mean(u**2):.6f} "
          f"m4={np.mean(u**4):.5f}  m6={np.mean(u**6):.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["reduce", "limit", "both"])
    a = ap.parse_args()
    if a.part in ("reduce", "both"):
        reduction(np.random.default_rng(20260904))
    if a.part in ("limit", "both"):
        limit()
