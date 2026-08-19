#!/usr/bin/env python3
"""
generator_template_v1_0_0.py
============================
Importable module + worked example for experiment generators, implementing
the VEE v2.3 fragility checks so they happen when the number is produced
rather than when the table is typeset.

WHAT IT ENFORCES
  - every reported quantity is tagged A / B / C / D (see VEE v2.3 sec.1);
  - a standard error is computed for each, analytically where a formula
    exists and by bootstrap otherwise;
  - displayed precision is DERIVED from that standard error, never assumed;
  - the whole experiment runs under two independent seeds and only digits
    agreeing between them are emitted, with disagreements flagged;
  - CSV is written with csv.writer (quoting, LF) and carries n, seed, class
    and standard error alongside every value;
  - the generator records the SHA-256 of its own source in the output.

USAGE
    from generator_template_v1_0_0 import Report, Q
    def experiment(seed):
        rng = np.random.default_rng(seed)
        ...
        return {"bias": Q(samples, "mean"),
                "variance": Q(samples, "variance"),
                "ratio": Q(num_samples, "ratio", denom=den_samples)}
    Report("my_experiment", experiment, seeds=(11, 12)).run().write("out.csv")

Run this file directly for a self-test on synthetic data.
"""

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

CLASS_OF = {"exact": "A", "mean": "B", "variance": "C",
            "ratio": "D", "difference": "D", "root": "D"}


class Q:
    """One reported quantity, with its fragility class and standard error."""

    def __init__(self, samples, kind, denom=None, slope=None, note=""):
        self.kind = kind
        self.cls = CLASS_OF[kind]
        self.note = note
        self.n = 1 if kind == "exact" else len(np.atleast_1d(samples))
        x = np.atleast_1d(np.asarray(samples, dtype=float))

        if kind == "exact":
            self.value, self.se = float(x[0]), 0.0

        elif kind == "mean":
            self.value = float(x.mean())
            self.se = float(x.std(ddof=1) / math.sqrt(self.n))

        elif kind == "variance":
            self.value = float(x.var(ddof=1))
            self.se = self.value * math.sqrt(2.0 / (self.n - 1))

        elif kind == "ratio":
            d = np.atleast_1d(np.asarray(denom, dtype=float))
            self.value = float(x.mean() / d.mean())
            self.se = self._bootstrap_ratio(x, d)

        elif kind == "difference":
            d = np.atleast_1d(np.asarray(denom, dtype=float))
            self.value = float(x.mean() - d.mean())
            self.se = float(math.hypot(x.std(ddof=1) / math.sqrt(len(x)),
                                       d.std(ddof=1) / math.sqrt(len(d))))

        elif kind == "root":
            # samples = the located root; slope = |g'| at it; denom = sigma(g)
            if slope is None or denom is None:
                raise ValueError("kind='root' needs slope=|g'| and denom=sigma(g)")
            self.value = float(x[0])
            self.se = float(denom) / abs(float(slope))

        else:
            raise ValueError(kind)

    @staticmethod
    def _bootstrap_ratio(x, d, draws=2000, seed=0):
        g = np.random.default_rng(seed)
        r = [x[g.integers(0, len(x), len(x))].mean()
             / d[g.integers(0, len(d), len(d))].mean() for _ in range(draws)]
        return float(np.std(r, ddof=1))

    # ---- presentation ------------------------------------------------
    def sig_digits(self):
        """Digits supported: last shown digit sits at the order of the SE."""
        if self.se <= 0 or self.value == 0:
            return 6
        return max(1, int(math.floor(math.log10(abs(self.value))))
                   - int(math.floor(math.log10(self.se))) + 1)

    def display(self):
        k = self.sig_digits()
        if self.se <= 0:
            return f"{self.value:.{max(k-1,0)}e}"
        return f"{self.value:.{k-1}e} +/- {self.se:.1e}"

    def digits_str(self):
        k = self.sig_digits()
        return f"{self.value:.{k-1}e}"


class Report:
    """Runs an experiment under two seeds and keeps only agreeing digits."""

    def __init__(self, name, experiment, seeds=(11, 12), source=None):
        self.name, self.experiment, self.seeds = name, experiment, seeds
        self.source = Path(source or __file__)
        self.rows, self.flags = [], []

    def run(self):
        a = self.experiment(self.seeds[0])
        b = self.experiment(self.seeds[1])
        assert a.keys() == b.keys(), "the two seeds reported different quantities"
        for key in a:
            qa, qb = a[key], b[key]
            shown, agree = self._agreeing(qa, qb)
            if agree < qa.sig_digits():
                self.flags.append(
                    f"{key}: SE supports {qa.sig_digits()} digit(s) but the two "
                    f"seeds agree on {agree} ({qa.digits_str()} vs {qb.digits_str()}); "
                    f"showing {agree}")
            self.rows.append(dict(
                quantity=key, value=shown, se=f"{qa.se:.2e}",
                fragility_class=qa.cls, kind=qa.kind, n=qa.n,
                digits_shown=agree, digits_from_se=qa.sig_digits(),
                seed_a=self.seeds[0], seed_b=self.seeds[1], note=qa.note))
        return self

    @staticmethod
    def _agreeing(qa, qb):
        """Longest prefix of significant digits identical under both seeds."""
        cap = min(qa.sig_digits(), qb.sig_digits())
        best = 0
        for k in range(1, cap + 1):
            if f"{qa.value:.{k-1}e}" == f"{qb.value:.{k-1}e}":
                best = k
            else:
                break
        best = max(best, 1)
        return f"{qa.value:.{best-1}e}", best

    def write(self, path):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()[:16]
        with open(path, "w", newline="") as fh:
            # csv.writer defaults to lineterminator='\r\n' whatever open() is
            # told; the terminator must be set on the writer itself.
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["quantity", "value", "se", "fragility_class", "kind",
                        "n", "digits_shown", "digits_from_se", "seed_a",
                        "seed_b", "note", "generator_sha16"])
            for r in self.rows:
                w.writerow([r["quantity"], r["value"], r["se"],
                            r["fragility_class"], r["kind"], r["n"],
                            r["digits_shown"], r["digits_from_se"],
                            r["seed_a"], r["seed_b"], r["note"], digest])
        print(f"wrote {path}  (generator sha16 {digest})")
        for f in self.flags:
            print(f"  FLAG  {f}")
        heavy = [r for r in self.rows if r["fragility_class"] == "D"]
        if heavy:
            print("  VEE-F1: Class D present. Confirm no conclusion rests on "
                  "these:", ", ".join(r["quantity"] for r in heavy))
        return self


# ---------------------------------------------------------------- self-test
def _demo(seed):
    """Synthetic stand-in reproducing the session's failure modes."""
    rng = np.random.default_rng(seed)
    a = rng.normal(3.0, 0.1, 100)      # a well-behaved mean
    b = rng.normal(3.0, 0.1, 60)       # a second, for a fragile ratio
    return {
        "mean_estimate": Q(a, "mean"),
        "variance_estimate": Q(a, "variance"),
        "variance_ratio": Q(a, "ratio", denom=b, note="Class D by construction"),
        "equilibrium": Q([3.0 * 2048 / 2050], "exact", note="3N/(N+2)"),
        "crossing": Q([0.318], "root", slope=3.7, denom=0.045,
                      note="f* from a C/Cbar crossing"),
    }


if __name__ == "__main__":
    print("self-test on synthetic data\n" + "-" * 60)
    rep = Report("demo", _demo, seeds=(11, 12)).run()
    for r in rep.rows:
        print(f"  {r['quantity']:>18}  {r['value']:>12}  +/- {r['se']:>9}  "
              f"class {r['fragility_class']}  n={r['n']:<4} "
              f"digits {r['digits_shown']}/{r['digits_from_se']}")
    print()
    rep.write("generator_template_selftest.csv")
