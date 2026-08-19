#!/usr/bin/env python3
"""Manuscript build verifier — Phase A instrument.

Replaces the ad-hoc grep suite quoted about twelve times this session, which
reports a perfect compile against a truncated or absent log because every one
of its pass conditions is satisfiable by the absence of data.

Three rules, applied to every check here:

  INDEPENDENCE  the expectation comes from a different source than the
                observation. Page count is read from the log AND from the PDF
                itself. Bibliography counts come from .aux, .bib and .bbl,
                three files written by three different programs.
  DENOMINATOR   every check reports what it examined, not only what it found.
                "0 errors" is unfalsifiable; "0 errors in 495 lines" is not.
  REFUSAL       a check that cannot see enough input REFUSES rather than
                passing. Silence is not success.

Run --selftest first. It feeds each check deliberately broken input and
requires the check to fail; if any check passes on garbage, the whole
instrument reports itself untrustworthy and exits nonzero.

Usage:
    python3 verify_manuscript.py --dir BUILD --stem hyman2026obstruction_v1_19_0
    python3 verify_manuscript.py --selftest
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

MIN_LOG_LINES = 50          # a real pdflatex log is hundreds of lines
MIN_TEX_BYTES = 5000


class Refused(Exception):
    """Raised when a check cannot see enough input to mean anything."""


def _read(path, min_bytes=1):
    if not os.path.exists(path):
        raise Refused(f"{os.path.basename(path)} does not exist")
    data = open(path, "rb").read()
    if len(data) < min_bytes:
        raise Refused(f"{os.path.basename(path)} is {len(data)} bytes, expected >= {min_bytes}")
    return data.decode("utf-8", errors="replace")


def check_log(logpath):
    """Errors, undefined references, warnings, overfull boxes — all with denominators."""
    text = _read(logpath)
    lines = text.splitlines()
    if len(lines) < MIN_LOG_LINES:
        raise Refused(f"log has {len(lines)} lines, expected >= {MIN_LOG_LINES}; nothing verified")
    if "Output written" not in text and "No pages of output" not in text:
        raise Refused("log records no output stage; the build did not finish")
    n = len(lines)
    return {
        "log lines scanned": n,
        "errors": (sum(1 for l in lines if l.startswith("!")), n),
        "undefined": (sum(1 for l in lines if "undefined" in l.lower()), n),
        "latex warnings": (sum(1 for l in lines if "LaTeX Warning" in l), n),
        "package warnings": (len(re.findall(r"Package .* Warning", text)), n),
        "overfull hboxes": (sum(1 for l in lines if l.startswith("Overfull \\hbox")), n),
    }


def check_pages(logpath, pdfpath):
    """INDEPENDENCE: page count from the log and from the PDF must agree."""
    text = _read(logpath)
    m = re.search(r"Output written on \S+ \((\d+) pages", text)
    if not m:
        raise Refused("log states no page count")
    from_log = int(m.group(1))
    if not os.path.exists(pdfpath):
        raise Refused("PDF absent; cannot cross-check the page count")
    out = subprocess.run(["pdfinfo", pdfpath], capture_output=True, text=True)
    m2 = re.search(r"^Pages:\s+(\d+)", out.stdout, re.M)
    if not m2:
        raise Refused("pdfinfo reports no page count")
    from_pdf = int(m2.group(1))
    if from_log != from_pdf:
        raise Refused(f"page count disagrees: log {from_log}, pdf {from_pdf}")
    return {"pages (log == pdf)": from_log}


def check_bib(auxpath, bibpath, bblpath):
    """INDEPENDENCE: three counts from three files written by three programs."""
    aux, bib, bbl = _read(auxpath), _read(bibpath), _read(bblpath)
    cited = set()
    for m in re.findall(r"\\citation\{([^}]*)\}", aux):
        cited.update(k.strip() for k in m.split(",") if k.strip())
    defined = len(re.findall(r"^@", bib, re.M))
    resolved = len(re.findall(r"\\bibitem", bbl))
    if not cited:
        raise Refused("aux records no citations")
    if len(cited) != defined or defined != resolved:
        raise Refused(f"bibliography disagrees: cited {len(cited)}, defined {defined}, resolved {resolved}")
    return {"bib cited == defined == resolved": len(cited)}


def check_dashes(texpaths):
    """Byte-level: literal ---, U+2014, U+2013. Missing file raises, not passes."""
    tot = {"---": 0, "U+2014": 0, "U+2013": 0}
    scanned = 0
    for p in texpaths:
        if not os.path.exists(p):
            raise Refused(f"{os.path.basename(p)} absent; dash scan means nothing")
        d = open(p, "rb").read()
        if len(d) < 100:
            raise Refused(f"{os.path.basename(p)} is {len(d)} bytes")
        scanned += len(d)
        tot["---"] += d.count(b"---")
        tot["U+2014"] += d.count(b"\xe2\x80\x94")
        tot["U+2013"] += d.count(b"\xe2\x80\x93")
    return {"bytes scanned": scanned, "em/en dashes": (sum(tot.values()), scanned)}


def run(build, stem, extra_tex=()):
    j = lambda e: os.path.join(build, stem + e)
    results, refusals = {}, []
    for name, fn in [
        ("log", lambda: check_log(j(".log"))),
        ("pages", lambda: check_pages(j(".log"), j(".pdf"))),
        ("bib", lambda: check_bib(j(".aux"), os.path.join(build, stem.rsplit("_v",1)[0] + ".bib"), j(".bbl"))),
        ("dashes", lambda: check_dashes([j(".tex")] + [os.path.join(build, f) for f in extra_tex])),
    ]:
        try:
            results[name] = fn()
        except Refused as e:
            refusals.append(f"{name}: REFUSED — {e}")
    return results, refusals


def report(results, refusals):
    for grp, vals in results.items():
        print(f"  [{grp}]")
        for k, v in vals.items():
            if isinstance(v, tuple):
                print(f"      {k:<24} {v[0]} of {v[1]} examined")
            else:
                print(f"      {k:<24} {v}")
    for r in refusals:
        print(f"  {r}")
    bad = sum(v[0] for vals in results.values() for v in vals.values()
              if isinstance(v, tuple))
    ok = (not refusals) and bad == 0
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}"
          f"  ({len(refusals)} refusals, {bad} defects found)")
    return ok


def selftest():
    """Every check must FAIL on broken input. If one passes, the tool is fake."""
    print("SELFTEST — each check is fed broken input and must refuse\n")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        cases = []
        # absent log
        cases.append(("absent log", lambda: check_log(os.path.join(td, "nope.log"))))
        # truncated log
        short = os.path.join(td, "short.log")
        open(short, "w").write("\n".join(f"line {i}" for i in range(10)))
        cases.append(("truncated log", lambda: check_log(short)))
        # log with no output stage
        noout = os.path.join(td, "noout.log")
        open(noout, "w").write("\n".join(f"line {i}" for i in range(200)))
        cases.append(("log with no output stage", lambda: check_log(noout)))
        # page count present in log, PDF absent
        pg = os.path.join(td, "pg.log")
        open(pg, "w").write("Output written on x.pdf (18 pages, 1 bytes).\n" * 60)
        cases.append(("pdf absent for cross-check", lambda: check_pages(pg, os.path.join(td, "nope.pdf"))))
        # empty aux
        for f, c in (("e.aux", ""), ("e.bib", "@x{y,}"), ("e.bbl", "\\bibitem{y}")):
            open(os.path.join(td, f), "w").write(c)
        cases.append(("aux with no citations",
                      lambda: check_bib(os.path.join(td, "e.aux"), os.path.join(td, "e.bib"),
                                        os.path.join(td, "e.bbl"))))
        # absent tex for dash scan
        cases.append(("absent tex for dash scan", lambda: check_dashes([os.path.join(td, "nope.tex")])))
        for label, fn in cases:
            try:
                fn()
                print(f"  {label:<32} PASSED ON GARBAGE  <-- CHECK IS FAKE")
                ok = False
            except Refused as e:
                print(f"  {label:<32} refused correctly ({str(e)[:44]})")
    print(f"\nSELFTEST: {'PASS — instrument is trustworthy' if ok else 'FAIL — do not trust this tool'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--stem")
    ap.add_argument("--extra", nargs="*", default=["phasef_tables.tex"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        sys.exit("Aborting: instrument failed its own self-test.")
    print(f"\nVERIFYING {a.stem} in {a.dir}\n")
    sys.exit(0 if report(*run(a.dir, a.stem, a.extra)) else 1)
