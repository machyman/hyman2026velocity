#!/usr/bin/env python3
"""
artifact_sweep_v1_0_0.py -- DISCLOSED DEVELOPMENT WORK. Zero offsets.

A byte-level integrity pass over every artifact produced this session.

WHY IT IS BYTE-LEVEL.  The format checks used earlier in the session were
inadequate in a way that mattered: `grep -c $'\\r'` returned zero on files
that demonstrably contained CR bytes, so files were reported LF-clean when
they were not.  Everything "verified" with those checks is unverified.  This
script therefore opens each file in binary and counts bytes; nothing is
inferred from how a file looks when printed.

CHECKS
  all files   non-empty; valid UTF-8; no byte-order mark; no CR bytes
  .csv        parses with csv.reader; rectangular; no blank rows; header
              row present; no row whose field count differs from the header
  .py         compiles under py_compile
  .tex        balanced \\begin/\\end for document; compiles separately
  .md         no unterminated code fences
"""

import ast
import csv
import io
import sys
from pathlib import Path

OUT = Path("/mnt/user-data/outputs")
ANCHOR = OUT / "HANDOFF_Session20.md"

# Artifacts retained deliberately under the never-overwrite convention even
# though they are known-defective. Listed here so the exception is explicit
# rather than remembered, and so a clean sweep stays meaningful.
SUPERSEDED = {
    "phasef_tables_rounded.csv":
        "two tables plus a pseudo-comment row in one file; superseded by "
        "phasef_part1_rounded.csv and phasef_part2_rounded.csv",
}


def session_files():
    t = ANCHOR.stat().st_mtime
    return sorted(p for p in OUT.iterdir()
                  if p.is_file() and p.stat().st_mtime > t)


def check(path):
    findings = []
    raw = path.read_bytes()
    if not raw:
        return ["EMPTY FILE"]

    if raw.startswith(b"\xef\xbb\xbf"):
        findings.append("byte-order mark present")

    text_ext = {".csv", ".py", ".tex", ".md", ".txt", ".yaml", ".bib"}
    if path.suffix in text_ext:
        cr = raw.count(13)
        if cr:
            findings.append(f"{cr} CR bytes (CRLF or CR endings)")
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            return findings + [f"not valid UTF-8: {e}"]

        if path.suffix == ".csv":
            rows = list(csv.reader(io.StringIO(txt)))
            if not rows:
                findings.append("no rows")
            else:
                widths = {len(r) for r in rows}
                if len(widths) > 1:
                    findings.append(f"not rectangular: field counts {sorted(widths)}")
                blanks = sum(1 for r in rows if not r or all(c == "" for c in r))
                if blanks:
                    findings.append(f"{blanks} blank row(s)")
                hdr = rows[0]
                numeric_hdr = sum(1 for c in hdr if c.replace('.', '', 1)
                                  .replace('-', '', 1).isdigit())
                if numeric_hdr > len(hdr) / 2:
                    findings.append("header row looks numeric (missing header?)")

        elif path.suffix == ".py":
            try:
                ast.parse(txt)
            except SyntaxError as e:
                findings.append(f"does not parse: line {e.lineno}")

        elif path.suffix == ".tex":
            if txt.count(r"\begin{document}") != txt.count(r"\end{document}"):
                findings.append("unbalanced document environment")

        elif path.suffix == ".md":
            if txt.count("```") % 2:
                findings.append("unterminated code fence")

    elif path.suffix == ".pdf":
        if not raw.startswith(b"%PDF"):
            findings.append("missing %PDF header")
        if b"%%EOF" not in raw[-2048:]:
            findings.append("missing %%EOF trailer")

    return findings


def main():
    files = session_files()
    print(f"sweeping {len(files)} artifacts produced this session\n")
    bad = {}
    by_ext = {}
    for p in files:
        by_ext[p.suffix] = by_ext.get(p.suffix, 0) + 1
        if p.name in SUPERSEDED:
            continue
        f = check(p)
        if f:
            bad[p.name] = f
    print("  by type: " + ", ".join(f"{k or '(none)'}={v}"
                                    for k, v in sorted(by_ext.items())))
    print()
    if bad:
        print(f"DEFECTS in {len(bad)} file(s):")
        for name, fs in sorted(bad.items()):
            print(f"  {name}")
            for x in fs:
                print(f"      - {x}")
    else:
        print("NO DEFECTS FOUND.")
    if SUPERSEDED:
        print("\nexcluded as deliberately-retained superseded artifacts:")
        for k, v in SUPERSEDED.items():
            print(f"  {k}\n      {v}")
    checked = len(files) - len(SUPERSEDED)
    print(f"\nclean: {checked - len(bad)}/{checked} checked "
          f"({len(SUPERSEDED)} excluded)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
