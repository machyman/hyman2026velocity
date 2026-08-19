#!/usr/bin/env python3
"""Scan for sentences claiming more scope than Definition 3.2 supports.

Mechanical enumeration; disposition is judged by hand. Written after three
consecutive fix-one-miss-the-twin failures, on the principle that reading for
these is the step that keeps failing.
"""
import re, sys
PATTERNS = [
    r"cannot work", r"no choice of sorting", r"no sorting key", r"any sorting key",
    r"Array-RQMC requires", r"requires a state-dependent sort", r"repairs the method",
    r"impossib", r"for this problem class", r"the method for", r"no key",
    r"rules out", r"in general\b", r"\bnever\b.{0,40}(work|preserve|repair)",
]
src = open(sys.argv[1], encoding="utf-8").read().splitlines()
hits = 0
for i, line in enumerate(src, 1):
    if line.lstrip().startswith("%"):
        continue
    for p in PATTERNS:
        if re.search(p, line, re.I):
            ctx = " ".join(src[max(0, i-2):i+1]).strip()
            print(f"  L{i:<5} [{p}]\n         …{ctx[-165:]}")
            hits += 1
            break
print(f"\n  {hits} candidate sentences in {len(src)} lines scanned")
