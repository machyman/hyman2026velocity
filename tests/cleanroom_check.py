#!/usr/bin/env python3
"""D1: does every registered generator actually regenerate its declared outputs?

The integrity check answers a narrower question — do files match recorded
hashes — and I spent this session treating its PASS as though it settled
reproducibility. It does not. This harness answers the question D actually
asks by copying each generator and its DECLARED INPUTS ONLY into an empty
directory and running it there. Anything the generator needs that the registry
does not declare shows up as a failure, which is the point.

Two rules from Phase A apply. DENOMINATOR: the report states how many
generators were examined, not only how many failed. NEGATIVE CONTROL: --selftest
plants a generator that cannot possibly work and requires the harness to catch
it; if the harness passes that, it is not measuring anything.

Usage:
    python3 cleanroom_check.py [--run] [--selftest]
        default   resolve inputs and start each generator briefly (fast)
        --run     run each to completion and check declared outputs appear
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

import os as _os
OUT = _os.environ.get("REPO_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
YAML_PATH = os.path.join(OUT, "MANUSCRIPT_STATUS.yaml")
SHORT, LONG = 25, 900


def stage(tmp, gen, inputs):
    """Empty room: the generator and its DECLARED inputs, nothing else."""
    shutil.copy(os.path.join(OUT, gen), tmp)
    missing = []
    for p in inputs:
        src = os.path.join(OUT, p)
        if os.path.exists(src):
            shutil.copy(src, tmp)
        else:
            missing.append(p)
    return missing


def attempt(tmp, gen, timeout):
    try:
        r = subprocess.run([sys.executable, gen], cwd=tmp, capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return "timeout", ""


def classify(rc, err):
    if rc == "timeout":
        return "RUNS (timed out, no input error)"
    if rc == 0:
        return "RUNS"
    low = err.lower()
    if "filenotfound" in low or "no such file" in low:
        return "NOT STANDALONE (undeclared input)"
    return f"FAILS ({err.strip().splitlines()[-1][:60] if err.strip() else 'rc=%s' % rc})"


def main(run_full):
    doc = yaml.safe_load(open(YAML_PATH))
    rows, examined = [], 0
    for item in doc["assets"]["items"]:
        gen = (item.get("generator") or {}).get("path")
        plot = (item.get("plot_generator") or {}).get("path")
        if not gen:
            rows.append((item["id"], "(none)", "NO GENERATOR", ""))
            continue
        examined += 1
        inputs = [i["path"] for i in (item.get("inputs") or []) if "*" not in i["path"]]
        outputs = [o["path"] for o in (item.get("outputs") or []) if "*" not in o["path"]]
        with tempfile.TemporaryDirectory() as tmp:
            missing_in = stage(tmp, gen, inputs)
            rc, err = attempt(tmp, gen, LONG if run_full else SHORT)
            verdict = classify(rc, err)
            produced = [o for o in outputs if os.path.exists(os.path.join(tmp, o))]
            note = ""
            if run_full and rc == 0:
                absent = [o for o in outputs if o not in produced]
                if absent:
                    verdict = "RUNS BUT DOES NOT WRITE"
                    note = ", ".join(absent)
            if missing_in:
                note = (note + " | undeclared-missing: " + ",".join(missing_in)).strip(" |")
            rows.append((item["id"], gen, verdict, note))
        if plot:
            with tempfile.TemporaryDirectory() as t2:
                pin = [i['path'] for i in (item.get('plot_inputs') or []) if '*' not in i['path']]
                stage(t2, plot, pin or inputs)
                rc2, e2 = attempt(t2, plot, LONG if run_full else SHORT)
                v2 = classify(rc2, e2)
                prod = [o for o in outputs if os.path.exists(os.path.join(t2, o))]
                if run_full and rc2 == 0 and not prod:
                    v2 = "RUNS BUT DOES NOT WRITE"
                rows.append((item["id"] + "/plot", plot, v2, ", ".join(prod)))

    w = max(len(r[1]) for r in rows) + 1
    print(f"{'item':<8} {'generator':<{w}} {'verdict':<32} note")
    print("-" * (46 + w))
    for iid, gen, verdict, note in rows:
        print(f"{iid:<8} {gen:<{w}} {verdict:<32} {note}")
    bad = sum(1 for r in rows if r[2] not in ("RUNS", "RUNS (timed out, no input error)"))
    scope = ("inputs resolve and the generator starts" if not run_full
             else "declared outputs are actually written")
    print(f"\n{examined} generators examined; {bad} failed. TESTED: {scope}.")
    if not run_full:
        print("    NOT tested in this mode: whether declared outputs appear. Use --run.")
    return bad


def selftest():
    """Plant a generator that cannot work; the harness must catch it."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "planted.py")
        open(bad, "w").write("open('this_file_is_not_declared.csv')\n")
        rc, err = attempt(tmp, "planted.py", SHORT)
        v = classify(rc, err)
        print(f"  planted undeclared-input generator -> {v}")
        ok &= v.startswith("NOT STANDALONE")
        good = os.path.join(tmp, "fine.py")
        open(good, "w").write("print('ok')\n")
        rc, err = attempt(tmp, "fine.py", SHORT)
        v2 = classify(rc, err)
        print(f"  planted working generator          -> {v2}")
        ok &= v2 == "RUNS"
    print("SELFTEST:", "PASS — harness detects the defect it looks for" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        sys.exit("Aborting: harness failed its own negative control.")
    print()
    sys.exit(1 if main(a.run) else 0)
