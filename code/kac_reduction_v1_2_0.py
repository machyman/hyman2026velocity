#!/usr/bin/env python3
"""
kac_reduction_v1_2_0.py
=======================
KAC REDUCTION, PHASE E: does the energy-similarity mechanism make a
PARAMETER-FREE quantitative prediction, or only a sign prediction?
DISCLOSED DEVELOPMENT WORK. Zero campaign offsets (seed base 813000000).

Phase D established that within-pair energy SIMILARITY sets the sign of the
distortion: similar-speed pairing condenses (kurtosis >> 0), extremal-speed
pairing compresses (kurtosis < 0), random pairing is neutral.  r5 draft v2
promotes the sign flip to the decisive arm.  Before registering that, the
mechanism should be pushed until it either breaks or yields something
sharper than a sign.

The sharper claim available is a ZERO CROSSING.  Introduce a one-parameter
family of pairing rules that interpolates continuously from similar to
dissimilar:

    'sep:d'  -- sort by |v|; cut the sorted ranks into consecutive blocks of
                size 2d; inside each block pair rank j with rank j+d.
                d = 1     -> adjacent ranks (maximally similar speeds)
                d = N/2   -> one block, lower half paired to upper half
                             (maximally dissimilar within this family)

and a rule-intrinsic descriptor of how dissimilar the pairs are:

    S(d) = E|v_a^2 - v_b^2| under the rule / E|v_a^2 - v_b^2| under uniform
           random matching, evaluated on the same state.

S = 0 is perfect energy matching, S = 1 is what random pairing delivers,
S > 1 is anti-matched.  The mechanism says the Maxwellian is stationary
exactly when the pairing is exchangeable, and that the sign of the
distortion tracks energy similarity.  If the mechanism is right and S is the
controlling statistic, then excess kurtosis must cross zero AT S = 1 -- with
no fitted parameter.  That is a real risk: the crossing could land anywhere,
or kurtosis could fail to be a monotone function of S at all.

PASS  -> r5's decisive arm upgrades from a sign test to a crossing test.
FAIL  -> S is not the controlling statistic; Item 2 should be softened and
         E-anti registered as a sign test only, or demoted.
"""

import argparse
import time

import numpy as np

from kac_reduction_v1_0_0 import (
    V_TH, DT, N_STEPS, RED_SEED_BASE,
    stratified_unit_sample, mmd_to_maxwellian, dv_plateau, state_diagnostics,
)
from kac_reduction_v1_1_0 import make_pairs as make_pairs_d

SNAP_STEPS_E = (0, 200, 400, 600, 800)      # t = 0, 10, 20, 30, 40


def make_pairs_e(v, pairing, rng):
    """Adds the rank-separation and blend families to the Phase D rules."""
    if pairing.startswith("blend:"):
        # Sufficiency probe: sort by |v|, send a random fraction f of the
        # particles to ADJACENT pairing among themselves (energy-matched,
        # S ~ 0) and the rest to EXTREMAL pairing among themselves
        # (energy-antimatched, S > 1).  f tunes S continuously through 1
        # while the rule stays strongly non-exchangeable throughout.  If S
        # were a sufficient statistic, the f giving S = 1 would give
        # kurtosis 0, as uniform random matching does.
        frac = float(pairing.split(":")[1])
        n_part = v.size
        order = np.argsort(np.abs(v), kind="stable")
        n_sim = int(round(frac * n_part)) // 2 * 2
        pick = rng.permutation(n_part)
        sim_pos = np.sort(pick[:n_sim])
        ext_pos = np.sort(pick[n_sim:])
        sim = order[sim_pos]
        ext = order[ext_pos]
        a = [sim[0::2]]; b = [sim[1::2]]
        half = ext.size // 2
        if half:
            a.append(ext[:half]); b.append(ext[::-1][:half])
        return np.concatenate(a), np.concatenate(b)

    if not pairing.startswith("sep:"):
        return make_pairs_d(v, pairing, rng)
    n_part = v.size
    sep = int(pairing.split(":")[1])
    sep = int(min(max(sep, 1), n_part // 2))
    order = np.argsort(np.abs(v), kind="stable")
    block = 2 * sep
    n_blocks = n_part // block
    head = order[: n_blocks * block].reshape(n_blocks, block)
    idx_a = head[:, :sep].ravel()
    idx_b = head[:, sep:].ravel()
    tail = order[n_blocks * block:]
    if tail.size >= 2:
        tail = rng.permutation(tail)
        half = tail.size // 2
        idx_a = np.concatenate([idx_a, tail[:half]])
        idx_b = np.concatenate([idx_b, tail[half: 2 * half]])
    return idx_a, idx_b


def energy_mismatch_ratio(v, pairing, rng, n_draws=8):
    """S = E|v_a^2 - v_b^2| under the rule, divided by the same under
    uniform random matching on the SAME state.  Rule-intrinsic, dimensionless."""
    idx_a, idx_b = make_pairs_e(v, pairing, rng)
    rule = float(np.mean(np.abs(v[idx_a] ** 2 - v[idx_b] ** 2)))
    ref = []
    for _ in range(n_draws):
        order = rng.permutation(v.size)
        ref.append(float(np.mean(np.abs(v[order[0::2]] ** 2 - v[order[1::2]] ** 2))))
    return rule / float(np.mean(ref))


def kac_step_e(v, pairing, rng, project=True):
    n_part = v.size
    idx_a, idx_b = make_pairs_e(v, pairing, rng)
    n_fired = idx_a.size
    theta = 2.0 * np.pi * stratified_unit_sample(n_fired, 1, rng)   # iid angles
    radius = np.hypot(v[idx_a], v[idx_b])
    v[idx_a] = radius * np.cos(theta)
    v[idx_b] = radius * np.sin(theta)
    if project:
        v -= v.mean()
        energy = np.dot(v, v)
        if energy > 0.0:
            v *= np.sqrt(n_part * V_TH * V_TH / energy)
    return v


def run_rep_e(n_part, pairing, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, V_TH, size=n_part)
    v -= v.mean()
    v *= np.sqrt(n_part * V_TH * V_TH / np.dot(v, v))
    plateau = dv_plateau(n_part)
    s_initial = energy_mismatch_ratio(v, pairing, np.random.default_rng(seed + 11))
    rows = []
    for step_idx in range(N_STEPS + 1):
        if step_idx:
            kac_step_e(v, pairing, rng)
        if step_idx in SNAP_STEPS_E:
            d_v = mmd_to_maxwellian(v)
            rows.append(dict(n_part=n_part, pairing=pairing, seed=seed,
                             t=step_idx * DT, s_initial=s_initial,
                             d_v=d_v, d_v_ratio=d_v / plateau,
                             **state_diagnostics(v)))
    return rows


def flush(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["crossing", "stability", "sufficiency"], default="crossing")
    ap.add_argument("--np", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    n_part, n_reps = args.np, args.reps
    t0 = time.time()

    if args.block == "crossing":
        seps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, n_part // 2]
        print(f"=== E1 ZERO-CROSSING TEST (Np={n_part}, {n_reps} reps, iid angles, t=40) ===")
        print(f"{'d':>7} {'S (energy mismatch)':>21} {'excess kurtosis':>18} {'D_v/plateau':>13}")
        rows = []
        for si, sep in enumerate(seps):
            pairing = f"sep:{sep}"
            for rep in range(n_reps):
                rows.extend(run_rep_e(n_part, pairing, RED_SEED_BASE + 300000 + 1000 * si + 10 * rep))
            sub = [r for r in rows if r["pairing"] == pairing and r["t"] == 40.0]
            s_val = float(np.mean([r["s_initial"] for r in sub]))
            kurt = float(np.mean([r["excess_kurtosis"] for r in sub]))
            sd = float(np.std([r["excess_kurtosis"] for r in sub], ddof=1))
            dv = float(np.exp(np.mean(np.log([r["d_v_ratio"] for r in sub]))))
            print(f"{sep:>7} {s_val:>21.4f} {kurt:>13.3f} +/-{sd:6.3f} {dv:>13.2f}")
        flush(rows, f"kac_crossing_Np{n_part}.csv")

    elif args.block == "sufficiency":
        fracs = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
        print(f"=== E3 SUFFICIENCY PROBE (Np={n_part}, {n_reps} reps, t=40) ===")
        print("    f = fraction of particles given energy-MATCHED (adjacent) pairing;")
        print("    the rest get energy-ANTIMATCHED (extremal) pairing.")
        print(f"{'f':>6} {'S (energy mismatch)':>21} {'excess kurtosis':>18} {'D_v/plateau':>13}")
        rows = []
        for fi, frac in enumerate(fracs):
            pairing = f"blend:{frac}"
            for rep in range(n_reps):
                rows.extend(run_rep_e(n_part, pairing, RED_SEED_BASE + 200000 + 1000 * fi + 10 * rep))
            sub = [r for r in rows if r["pairing"] == pairing and r["t"] == 40.0]
            s_val = float(np.mean([r["s_initial"] for r in sub]))
            kurt = float(np.mean([r["excess_kurtosis"] for r in sub]))
            sd = float(np.std([r["excess_kurtosis"] for r in sub], ddof=1))
            dv = float(np.exp(np.mean(np.log([r["d_v_ratio"] for r in sub]))))
            print(f"{frac:>6.2f} {s_val:>21.4f} {kurt:>13.3f} +/-{sd:6.3f} {dv:>13.2f}")
        flush(rows, f"kac_sufficiency_Np{n_part}.csv")

    else:
        print(f"=== E2 E-ANTI STABILITY (Np={n_part}, {n_reps} reps) ===")
        rows = []
        for rep in range(n_reps):
            rows.extend(run_rep_e(n_part, "absanti", RED_SEED_BASE + 400000 + 10 * rep))
        for t in (0.0, 10.0, 20.0, 30.0, 40.0):
            sub = [r for r in rows if r["t"] == t]
            kurt = np.mean([r["excess_kurtosis"] for r in sub])
            sd = np.std([r["excess_kurtosis"] for r in sub], ddof=1)
            dv = np.exp(np.mean(np.log([r["d_v_ratio"] for r in sub])))
            tail = np.mean([r["tail_frac_2vth"] for r in sub])
            print(f"  t={t:5.1f}  kurt {kurt:+8.4f} +/- {sd:.4f}   D_v/plateau {dv:7.2f}   tail {tail:.4f}")
        flush(rows, f"kac_eanti_stability_Np{n_part}.csv")

    print(f"  [wall clock {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
