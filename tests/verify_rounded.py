#!/usr/bin/env python3
"""Verify the hand-derived rounded CSVs against their generated sources.

Why a verifier and not a generator. The rounded files are not roundings: they
select columns, fix digits and add an n_reps column the source does not carry.
The derivation rule was never written down, so a generator would have to invent
one, and inventing a rule and calling it reproduction is worse than the gap it
replaces. What actually needs guarding is TRANSCRIPTION, and that is checkable
without knowing the rule: every value carried across must agree with the source
to the digits shown.

Phase A rules apply. INDEPENDENCE: the expected value comes from the generated
CSV, the observed from the hand-made one, two different files. DENOMINATOR: the
report says how many values were compared. NEGATIVE CONTROL: --selftest plants
a wrong digit and requires the checker to catch it.

Usage: python3 verify_rounded.py [--selftest]
"""
import argparse
import csv
import sys

import os
OUT = os.environ.get("REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# rounded column -> source column. n_reps is not in the source and is skipped.
MAPS = [
    ("phasef_part1.csv", "phasef_part1_rounded.csv", "scheme",
     {"bias_v4": "bias", "variance_v4": "variance", "mse_v4": "mse"}),
    ("phasef_part2.csv", "phasef_part2_rounded.csv", "w",
     {"c_over_cbar": "c_over_cbar", "bias_v4": "bias_v4", "mse_v4": "mse_v4",
      "bias_absv": "bias_absv", "var_absv": "var_absv", "mse_absv": "mse_absv"}),
]


def _resolve(name):
    """Repo layout groups data under data/; the flat working layout does not."""
    for c in (os.path.join(OUT, "data", name), os.path.join(OUT, name)):
        if os.path.exists(c):
            return c
    raise SystemExit(f"cannot find {name} under {OUT} or {OUT}/data")


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def sig(text):
    """Significant digits shown, e.g. '2.01e+03' -> 3, '0.991' -> 3."""
    m = text.lower().split("e")[0].lstrip("+-").replace(".", "").lstrip("0")
    return max(len(m.rstrip()), 1)


def agrees(shown, exact):
    """Does the exact value round to the shown value at the digits shown?"""
    try:
        s, x = float(shown), float(exact)
    except ValueError:
        return shown.strip() == exact.strip()
    if s == 0:
        return abs(x) < 5e-1
    from decimal import Decimal
    d = sig(shown)
    # round the exact value to d significant digits and compare as floats
    import math
    if x == 0:
        return abs(s) < 1e-30
    p = d - 1 - math.floor(math.log10(abs(x)))
    return round(x, p) == s


def run(rows_override=None):
    compared = bad = 0
    problems = []
    for src_name, rnd_name, key, cols in MAPS:
        src = {r[key]: r for r in load(_resolve(src_name))}
        rnd = rows_override.get(rnd_name) if rows_override else None
        rnd = rnd if rnd is not None else load(_resolve(rnd_name))
        for r in rnd:
            s = src.get(r[key])
            if s is None:
                problems.append(f"{rnd_name}: key {r[key]!r} absent from {src_name}")
                bad += 1
                continue
            for rc, sc in cols.items():
                compared += 1
                if not agrees(r[rc], s[sc]):
                    bad += 1
                    problems.append(f"{rnd_name} {key}={r[key]} {rc}: shows {r[rc]}, "
                                    f"source {src_name} has {s[sc]}")
    return compared, bad, problems


def report():
    compared, bad, problems = run()
    for p in problems:
        print("  MISMATCH:", p)
    print(f"\n  {compared} values compared against their generated sources, {bad} mismatched")
    print(f"  VERDICT: {'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


def selftest():
    print("SELFTEST — a wrong digit is planted and must be caught")
    rows = load(_resolve("phasef_part2_rounded.csv"))
    good_c, good_bad, _ = run()
    rows[0]["c_over_cbar"] = "9.99"
    _, bad, probs = run({"phasef_part2_rounded.csv": rows})
    ok = good_bad == 0 and bad >= 1
    print(f"  clean input  -> {good_bad} mismatches in {good_c} comparisons")
    print(f"  planted 9.99 -> {bad} mismatch(es): {probs[0][:70] if probs else 'NONE'}")
    print("SELFTEST:", "PASS — the checker detects a wrong digit"
          if ok else "FAIL — do not trust this checker")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        sys.exit("Aborting: checker failed its negative control.")
    print()
    sys.exit(0 if report() else 1)
