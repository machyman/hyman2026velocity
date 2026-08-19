#!/usr/bin/env python3
"""Build Colab wrapper notebooks for the companion repository.

Each notebook wraps one deposited generator so a reader can execute it in Colab
without installing anything. The notebooks are BUILT, not hand-written, so that
the eleven of them stay consistent and can be regenerated when a generator
changes. Hand-maintaining eleven near-identical notebooks is how they drift.

Structure follows the colab-notebook skill, research mode with companion-paper
cross-references: title cell, setup cell with SEED and a FULL/QUICK scale
switch, a verification suite that runs before any expensive computation, the
run cell, and a download cell last, which the skill marks mandatory.

Usage: python3 build_colab_wrappers.py --outdir notebooks/
"""
import argparse
import json
import os

REPO = "machyman/hyman2026velocity"
PAPER = ("Velocity sorting biases the equilibrium in Array-RQMC "
         "for binary-collision models")

# generator -> (order, short name, what it produces, paper cross-reference,
#               runtime, needs the campaign archive?)
NOTEBOOKS = [
    ("kac_onestep_test_v1_0_0.py", "01", "one_step_test",
     "Table 1, the direct one-step test of Lemma 4.2",
     "Lemma 4.2, Table 1", "~1 min", False),
    ("kac_onestep_curve_v1_0_0.py", "02", "one_step_curve",
     "the curve-rule row of the one-step test, on the applied particle loads",
     "Lemma 4.2, Section 7", "~1 min", True),
    ("kac_fig1_spectrum_v1_0_0.py", "03", "figure1_mechanism",
     "Figure 1, the C-spectrum over all matchings of one state",
     "Lemmas 4.1, 4.2 and 4.4; Figure 1", "~10 s", False),
    ("kac_manuscript_data_v1_0_0.py", "04", "figure2_data",
     "the finite-N sweep behind Figure 2, and the Section 5 counterexample",
     "Figure 2, Section 5", "~1 min", False),
    ("kac_fig2_finiteN_v1_0_0.py", "05", "figure2_plot",
     "Figure 2 itself, from the sweep CSV",
     "Figure 2", "~10 s", False),
    ("kac_fig3_tradeoff_v1_0_0.py", "06", "figure3_plot",
     "Figure 3, the trade-off across block width and the diagnostic",
     "Figure 3, Section 6", "~10 s", False),
    ("kac_phasef_v2_1_0.py", "07", "tables3_4",
     "Tables 3 and 4, the candidate fix and the block-width trade-off",
     "Tables 3 and 4", "~3 min", False),
    ("kac_remaining_v1_0_0.py", "08", "tables2_5",
     "Table 2, the moment ladder, and Table 5, the applied instance",
     "Tables 2 and 5, Sections 7 and 8", "~30 s", True),
]

BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         "(https://colab.research.google.com/github/{repo}/blob/main/notebooks/{fn})")


_N = [0]


def _id():
    """nbformat >= 4.5 requires a cell id; omitting it is a future hard error."""
    _N[0] += 1
    return f"cell{_N[0]:03d}"


def md(src):
    return {"cell_type": "markdown", "id": _id(), "metadata": {},
            "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "id": _id(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": src.splitlines(True)}


def build(gen, num, short, produces, xref, runtime, needs_archive):
    fn = f"{num}_{short}.ipynb"
    cells = []

    cells.append(md(f"""# {produces[0].upper() + produces[1:]}

{BADGE.format(repo=REPO, fn=fn)}

**Purpose.** Run `{gen}` from the companion repository and reproduce
{produces}.

**Companion paper.** {PAPER}
**Reproduces.** {xref}
**Expected runtime.** {runtime} on a free Colab CPU. No GPU is used.

**What this notebook is.** A wrapper. The science is in `{gen}`, which is
deposited unchanged in the repository; this notebook fetches it, checks the
environment, runs it, and hands you the outputs. Reading the generator is the
point, and it is printed below before it runs.
"""))

    cells.append(md("## Setup\n\nFetch the repository and pin the seed."))
    cells.append(code(f'''# ── Dependencies ──────────────────────────────────────────────────────────
"""
{produces[0].upper() + produces[1:]}
{'=' * 68}
Wrapper around {gen} from the companion repository for
"{PAPER}".

Reproduces: {xref}

Author:  James M. Hyman
         Department of Mathematics, Tulane University
         mhyman@tulane.edu
Date:    2026-08-16  Version 1.0
"""
import os, subprocess, sys, textwrap

REPO_URL = "https://github.com/{REPO}.git"
GENERATOR = "{gen}"

if not os.path.isdir("hyman2026velocity"):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
os.chdir("hyman2026velocity/{'data' if needs_archive else 'code'}")
print("working directory:", os.getcwd())

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-r", "../requirements.txt"], check=True)

# ── Reproducibility ───────────────────────────────────────────────────────
# The generator sets its own seeds internally; they are printed in its output
# and recorded in MANUSCRIPT_STATUS.yaml. Nothing here overrides them, because
# reproducing the paper's numbers means using the paper's seeds.
SEED = None

# ── Scale switch ──────────────────────────────────────────────────────────
# FULL = True  → the parameters that produced the published numbers
# FULL = False → not offered here: this generator has one published setting,
#                and a reduced run would not reproduce the paper.
FULL = True
print(f"Scale: {{'FULL (reproduces the published numbers)' if FULL else 'QUICK'}}")
print(f"Expected runtime: {runtime}")'''))

    cells.append(md("## Verification suite\n\nRun before the computation, so a "
                    "broken environment fails in seconds rather than minutes."))
    arch = '''
# Test 3: the campaign archive this generator needs is present
assert os.path.exists("discrepancy_campaign_R4_FULL_results_2026-08-12.zip"), \\
    "FAIL: the registered particle-load archive is missing"
print("Test 3 PASS  registered particle-load archive present")
N_TESTS = 3''' if needs_archive else "\nN_TESTS = 2"
    cells.append(code(f'''# ── Verification Suite ────────────────────────────────────────────────────
print("=" * 65); print("VERIFICATION SUITE"); print("=" * 65)

# Test 1: the generator is where the repository says it is
src = GENERATOR if os.path.exists(GENERATOR) else os.path.join("..", "code", GENERATOR)
assert os.path.exists(src), f"FAIL: {{GENERATOR}} not found"
print(f"Test 1 PASS  {{GENERATOR}} present, {{os.path.getsize(src)}} bytes")

# Test 2: numpy imports and agrees with a hand-checkable value.
# E[cos^4 t] = 3/8 for t uniform is the constant Lemma 4.2 turns on.
import numpy as np
t = np.linspace(0, 2 * np.pi, 2_000_001)
val = np.trapezoid(np.cos(t) ** 4, t) / (2 * np.pi)
assert abs(val - 3 / 8) < 1e-6, f"FAIL: E[cos^4] = {{val}}, expected 0.375"
print(f"Test 2 PASS  E[cos^4 t] = {{val:.6f}} = 3/8  (the constant in Lemma 4.2)"){arch}

print(f"\\nAll {{N_TESTS}} verification tests PASSED.")
print("=" * 65)'''))

    cells.append(md("## The generator\n\nPrinted before it runs. This is the "
                    "deposited file, unmodified."))
    cells.append(code('print(open(src).read())'))

    cells.append(md(f"## Run\n\nExecutes `{gen}` exactly as deposited."))
    cells.append(code(f'''# ── Run ───────────────────────────────────────────────────────────────────
import time
t0 = time.time()
env = dict(os.environ, SOURCE_DATE_EPOCH="1786822750")   # pins PDF timestamps
r = subprocess.run([sys.executable, src], capture_output=True, text=True, env=env)
print(r.stdout[-4000:])
if r.returncode != 0:
    print("STDERR:\\n", r.stderr[-2000:])
print(f"\\nexit code {{r.returncode}} in {{time.time() - t0:.1f}} s")
assert r.returncode == 0, "the generator did not complete"'''))

    cells.append(md("## Outputs"))
    cells.append(code('''# ── Outputs ───────────────────────────────────────────────────────────────
import glob
produced = sorted(set(glob.glob("*.csv") + glob.glob("*.pdf") + glob.glob("*.png")))
for f in produced:
    print(f"  {f}  ({os.path.getsize(f) // 1024} KB)")

for f in [p for p in produced if p.endswith(".png")]:
    from IPython.display import Image, display
    display(Image(f))'''))

    cells.append(md("## Download\n\nAlways the last cell."))
    cells.append(code('''# ── Download outputs ──────────────────────────────────────────────────────
output_files = produced
try:
    from google.colab import files
    for fname in output_files:
        files.download(fname)
    print("Downloads triggered.")
except ImportError:
    print("Not in Colab — files saved locally:")
    for fname in output_files:
        print(f"  {fname}  ({os.path.getsize(fname) // 1024} KB)")'''))

    return fn, {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="notebooks")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    for gen, num, short, produces, xref, runtime, arch in NOTEBOOKS:
        fn, nb = build(gen, num, short, produces, xref, runtime, arch)
        with open(os.path.join(a.outdir, fn), "w") as fh:
            json.dump(nb, fh, indent=1)
        rows.append((fn, gen, xref, runtime))
        print(f"  wrote {fn}")
    print(f"\n{len(rows)} notebooks. README table:\n")
    print("| Notebook | Wraps | Reproduces | Runtime |")
    print("|---|---|---|---|")
    for fn, gen, xref, rt in rows:
        print(f"| [`{fn}`](notebooks/{fn}) | `{gen}` | {xref} | {rt} |")


if __name__ == "__main__":
    main()
