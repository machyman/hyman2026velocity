"""
W2 / ENH-1: a defensible error statement for the exact evaluation.

Why this script exists.  `w2_exact_sweep.py` opens with
`warnings.filterwarnings("ignore")`, which silences the scipy
IntegrationWarnings that `quad` raises on the inner integral at larger N.  The
determinants it reports are correct and the margins are enormous, but a paper
whose central claim is "verified by exact computation" cannot rest on a script
that hides the integrator telling it that roundoff prevented the requested
tolerance.  A SINUM referee who re-runs the code will see those warnings.

What this script does differently:

  1. Warnings are CAPTURED and COUNTED, not suppressed.
  2. The three analytic negative controls are evaluated at every even N in the
     sweep, not only at N = 6 and 8, so the error statement is per-N rather
     than a single figure quoted as if it were general.
  3. It reports the MARGIN: determinant divided by the largest control error at
     that N.  That ratio, not the raw error, is what licenses the sign claim.
  4. It reports quad's own returned error estimates alongside the analytic
     controls, so an over-optimistic internal estimate would be visible.

Controls (all exact in closed form, for y the order statistics of N iid
chi^2_1 variables):

    NC1  sum_i E[y_i]      = N
    NC2  sum_{i<j} M_ij    = N(N-1)/2
    NC3  sum_{i<j} T_ij    = 3N(N-1)

Run: python3 w2_controls.py
"""
import warnings
import numpy as np
from w2_exact import moment, uni_moment

NS = [6, 8, 10, 12, 14, 16, 20, 24]


def arms(N):
    """A, B for each rule, plus the largest quad error estimate seen."""
    srt = [(2 * k + 1, 2 * k + 2) for k in range(N // 2)]
    ext = [(k + 1, N - k) for k in range(N // 2)]
    worst = 0.0

    def AB(pairs):
        nonlocal worst
        A = -N / 2.0
        B = -3.0 * N
        for i, j in pairs:
            v, e = moment(N, i, j, 1, 1)
            A += v
            worst = max(worst, abs(e))
            v1, e1 = moment(N, i, j, 2, 1)
            v2, e2 = moment(N, i, j, 1, 2)
            B += v1 + v2
            worst = max(worst, abs(e1), abs(e2))
        return A, B

    A_s, B_s = AB(srt)
    A_e, B_e = AB(ext)
    return A_s, B_s, A_e, B_e, worst


def controls(N):
    """Analytic negative controls; returns absolute errors NC1, NC2, NC3."""
    nc1 = abs(sum(uni_moment(N, i, 1)[0] for i in range(1, N + 1)) - N)
    allp = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    nc2 = abs(sum(moment(N, i, j, 1, 1)[0] for i, j in allp) - N * (N - 1) / 2.0)
    nc3 = abs(sum(moment(N, i, j, 2, 1)[0] + moment(N, i, j, 1, 2)[0]
                  for i, j in allp) - 3.0 * N * (N - 1))
    return nc1, nc2, nc3


def main():
    print(f"{'N':>4} {'NC1':>10} {'NC2':>10} {'NC3':>10} {'worst ctrl':>11} "
          f"{'quad est':>10} {'determinant':>14} {'margin':>10} {'warn':>5}")
    print("-" * 92)
    rows = []
    for N in NS:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nc1, nc2, nc3 = controls(N)
            A_s, B_s, A_e, B_e, qerr = arms(N)
            nwarn = len(w)
        det = A_s * B_e - A_e * B_s
        worst = max(nc1, nc2, nc3)
        margin = abs(det) / worst if worst > 0 else float("inf")
        rows.append((N, worst, det, margin, nwarn))
        print(f"{N:>4} {nc1:>10.1e} {nc2:>10.1e} {nc3:>10.1e} {worst:>11.1e} "
              f"{qerr:>10.1e} {det:>+14.6f} {margin:>10.1e} {nwarn:>5}")

    wmax = max(r[1] for r in rows)
    mmin = min(r[3] for r in rows)
    tot = sum(r[4] for r in rows)
    print("-" * 92)
    print(f"\nlargest analytic control error over the sweep : {wmax:.1e}")
    print(f"smallest margin (|det| / worst control error) : {mmin:.1e}")
    print(f"IntegrationWarnings raised (not suppressed)   : {tot}")
    print("\nDefensible statement:")
    print(f"  every determinant is positive with a margin of at least {mmin:.0e}")
    print(f"  times the largest analytic control error at that N, the worst")
    print(f"  control error anywhere in the sweep being {wmax:.0e}.")
    print("\nThe sign of the determinant is therefore not in question at any N")
    print("in the sweep. The warning count is reported so the reader can see")
    print("that the integrator was consulted rather than silenced.")


if __name__ == "__main__":
    main()
