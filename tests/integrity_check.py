#!/usr/bin/env python3
"""Integrity check for hyman2026obstruction MANUSCRIPT_STATUS.yaml.

Recomputes SHA-256 (first 16 hex chars, the convention used in this registry)
for every generator, input and output named in assets.items, plus the
bibliography and every protected_extra path.  Reports MATCH / MISMATCH /
MISSING / UNHASHED per path.

Self-test: run with --selftest to confirm the comparator actually detects a
planted mismatch and a planted missing file before any real verdict is trusted.
"""
import hashlib
import os
import sys

import yaml

import os as _os
OUT = _os.environ.get("REPO_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
YAML_PATH = os.path.join(OUT, "MANUSCRIPT_STATUS.yaml")


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def check_one(path, expected, role, item_id, rows):
    """Append one verdict row.  expected may be None (unhashed)."""
    full = path if os.path.isabs(path) else os.path.join(OUT, path)
    # Registry paths containing a glob or a parenthetical provenance note are
    # archive members, not loose files; they are handled separately.
    if "*" in path or "(" in path:
        rows.append((item_id, role, path, "ARCHIVE-MEMBER", expected, None))
        return
    if not os.path.exists(full):
        rows.append((item_id, role, path, "MISSING", expected, None))
        return
    actual = sha16(full)
    if expected is None:
        rows.append((item_id, role, path, "UNHASHED", None, actual))
    elif actual == expected:
        rows.append((item_id, role, path, "MATCH", expected, actual))
    else:
        rows.append((item_id, role, path, "MISMATCH", expected, actual))


def run(doc):
    rows = []
    bib = doc["manuscript"]["bibliography"]
    check_one(bib["bib_file"], bib["bib_sha256"], "bibliography", "bib", rows)

    for item in doc["assets"]["items"]:
        iid = item["id"]
        gen = item.get("generator") or {}
        if gen.get("path"):
            check_one(gen["path"], gen.get("sha256"), "generator", iid, rows)
        else:
            rows.append((iid, "generator", "(none recorded)", "UNHASHED", None, None))
        for inp in item.get("inputs") or []:
            check_one(inp["path"], inp.get("sha256"), "input", iid, rows)
        for outp in item.get("outputs") or []:
            check_one(outp["path"], outp.get("sha256"), "output", iid, rows)

    for p in doc["assets"].get("protected_extra") or []:
        full = os.path.join(OUT, p)
        rows.append((
            "protected", "extra", p,
            "PRESENT" if os.path.exists(full) else "MISSING",
            None, sha16(full) if os.path.exists(full) else None,
        ))
    return rows


def report(rows):
    w = max(len(r[2]) for r in rows) + 1
    print(f"{'item':<10} {'role':<12} {'path':<{w}} {'verdict':<15} expected -> actual")
    print("-" * (60 + w))
    for iid, role, path, verdict, exp, act in rows:
        tail = ""
        if verdict == "MISMATCH":
            tail = f"{exp} -> {act}"
        elif verdict == "UNHASHED" and act:
            tail = f"(null) -> {act}"
        elif verdict == "MATCH":
            tail = exp
        print(f"{iid:<10} {role:<12} {path:<{w}} {verdict:<15} {tail}")
    print()
    counts = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    print("TALLY:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    halting = counts.get("MISMATCH", 0) + counts.get("MISSING", 0)
    print("HALT-LEVEL FAILURES (mismatch+missing):", halting)
    print("INTEGRITY:", "PASS" if halting == 0 else "FAIL")
    return halting


def selftest():
    """Prove the comparator detects a mismatch and a missing file."""
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "good.txt")
        with open(good, "w") as fh:
            fh.write("hello")
        real = sha16(good)
        rows = []
        check_one(good, real, "t", "t", rows)
        ok &= rows[-1][3] == "MATCH"
        print(f"selftest correct-hash  -> {rows[-1][3]:<10} (expect MATCH)")
        rows = []
        check_one(good, "0" * 16, "t", "t", rows)
        ok &= rows[-1][3] == "MISMATCH"
        print(f"selftest planted-wrong -> {rows[-1][3]:<10} (expect MISMATCH)")
        rows = []
        check_one(os.path.join(td, "nope.txt"), "abc", "t", "t", rows)
        ok &= rows[-1][3] == "MISSING"
        print(f"selftest absent-file   -> {rows[-1][3]:<10} (expect MISSING)")
        rows = []
        check_one(good, None, "t", "t", rows)
        ok &= rows[-1][3] == "UNHASHED"
        print(f"selftest null-hash     -> {rows[-1][3]:<10} (expect UNHASHED)")
    print("SELFTEST:", "PASS — comparator is trustworthy" if ok else "FAIL — do not trust results")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        sys.exit("Aborting: comparator self-test failed.")
    print()
    with open(YAML_PATH) as fh:
        doc = yaml.safe_load(fh)
    sys.exit(1 if report(run(doc)) else 0)
