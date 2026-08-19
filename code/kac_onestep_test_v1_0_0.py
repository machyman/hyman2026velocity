#!/usr/bin/env python3
"""E1: direct one-step test of Lemma 4.2 / Theorem 4.6.
Criterion pre-registered in session_log.md BEFORE this run:
 (a) |empirical - predicted|/predicted < 0.5% for EVERY rule
 (b) discrepancy within 3 standard errors of zero
 (c) two seeds agree on reported digits
"""
import numpy as np
N, M, CHUNK = 2048, 4000, 1000

def states(m, rng):
    g = rng.normal(size=(m, N))
    return g * np.sqrt(N) / np.linalg.norm(g, axis=1, keepdims=True)

def pairs_for(rule, v, rng):
    m = v.shape[0]
    if rule == "fixed":
        idx = np.tile(np.arange(N), (m, 1))
    elif rule == "random":
        idx = np.argsort(rng.random((m, N)), axis=1)
    elif rule in ("sorted", "extremal"):
        idx = np.argsort(np.abs(v), axis=1)
        if rule == "extremal":
            idx = np.concatenate([idx[:, :N//2], idx[:, N//2:][:, ::-1]], axis=1)
            idx = idx.reshape(m, 2, N//2).transpose(0, 2, 1).reshape(m, N)
    else:                                   # block:w  -> sort, shuffle within blocks
        w = int(rule.split(":")[1])
        idx = np.argsort(np.abs(v), axis=1).reshape(m, N//w, w)
        idx = np.take_along_axis(idx, np.argsort(rng.random(idx.shape), axis=2), axis=2)
        idx = idx.reshape(m, N)
    return idx[:, 0::2], idx[:, 1::2]

def one_rule(rule, seed):
    rng = np.random.default_rng(seed)
    emp, pred = [], []
    for _ in range(M // CHUNK):
        v = states(CHUNK, rng)
        a, b = pairs_for(rule, v, rng)
        va, vb = np.take_along_axis(v, a, 1), np.take_along_axis(v, b, 1)
        R2 = va**2 + vb**2
        C = np.sum(va**2 * vb**2, axis=1)
        Q = np.sum(v**4, axis=1)
        th = rng.uniform(0, 2*np.pi, size=R2.shape)
        emp.append(np.sum(R2**2 * (np.cos(th)**4 + np.sin(th)**4), axis=1))
        pred.append(0.75 * (Q + 2*C))
    emp, pred = np.concatenate(emp), np.concatenate(pred)
    d = emp - pred
    return emp.mean(), pred.mean(), d.mean(), d.std(ddof=1)/np.sqrt(len(d))

import csv as _csv

def cbar_of(v):
    n = v.shape[1]; Q = np.sum(v**4, axis=1)
    return (n*n - Q) / (2*(n-1))

def with_ratio(rule, seed):
    """Same run, also returning C/Cbar, so every column comes from one run."""
    rng = np.random.default_rng(seed)
    v = states(CHUNK, rng)
    a, b = pairs_for(rule, v, rng)
    va, vb = np.take_along_axis(v, a, 1), np.take_along_axis(v, b, 1)
    return float(np.mean(np.sum(va**2 * vb**2, axis=1) / cbar_of(v)))

RULES = ["random", "fixed", "sorted", "extremal", "block:64", "block:512"]
print(f"{'rule':<10} {'empirical':>12} {'predicted':>12} {'rel.diff':>10} {'d/SE':>8} {'seed2 rel':>10}")
print("-"*68)
allpass = True
for r in RULES:
    e1, p1, d1, s1 = one_rule(r, 20260817)
    e2, p2, d2, s2 = one_rule(r, 77770817)
    rel1, rel2 = d1/p1, d2/p2
    z = d1/s1 if s1 else 0.0
    ok = abs(rel1) < 0.005 and abs(z) < 3 and round(rel1,5) == round(rel2,5)
    allpass &= ok
    print(f"{r:<10} {e1:>12.4f} {p1:>12.4f} {rel1:>+10.5%} {z:>8.2f} {rel2:>+10.5%} {'' if ok else '  <-- FAIL'}")
with open("onestep_test.csv", "w", newline="") as fh:
    w = _csv.writer(fh)
    w.writerow(["rule", "c_over_cbar", "predicted_EQ", "measured_EQ", "rel_diff", "d_over_SE"])
    for r in RULES:
        e, p, d, se = one_rule(r, 20260817)
        w.writerow([r, f"{with_ratio(r, 20260817):.3f}", f"{p:.4f}", f"{e:.4f}",
                    f"{d/p:+.5f}", f"{d/se:+.2f}"])
print("wrote onestep_test.csv")
print(f"\nCRITERION (a)(b)(c): {'MET for every rule' if allpass else 'NOT MET'}")
print("\nDEVIATION FROM PRE-REGISTRATION: the curve rule is omitted. It is defined")
print("on (position, velocity) pairs and requires a spatial load; a draw from")
print("sigma has no positions. 6 rules of the 7 registered were run.")
