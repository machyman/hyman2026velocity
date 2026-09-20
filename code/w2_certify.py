"""
W2c step 5: certification of the sign count, and the pencil.

Two questions.

(1) Is the sign count stable?  Refine the r-grid and re-derive.  Also measure
    how far Theta stays from zero away from its two roots, and check that the
    crossing at each root is transversal, so no extra pair of sign changes can
    be hiding inside a bracket.

(2) Does the same argument give MONOTONICITY, not just the N = 6 extremum?
    Anchoring at any even N0 instead of 6 gives Theta_{N0} = beta - S(N0)alpha,
    which still has the two vanishing moments m = 0 and m = N0-2.  So the
    interpolation argument applies verbatim with exponents (0, N0-2, m) and
    yields S(N) > S(N0) for N > N0 -- provided the pencil beta - lambda alpha
    has exactly two sign changes for every lambda in the range swept by S(N0).
"""
import sys
import numpy as np
from w2_grid import F_srt, F_ext, fA_vec, fB_vec, gl


def cache(which, nr):
    Ffun = F_srt if which == "srt" else F_ext
    r, wr = gl(nr)
    al = np.array([Ffun(fA_vec, x) for x in r])
    be = np.array([Ffun(fB_vec, x) for x in r])
    np.savez(f"cache_{which}_{nr}.npz", r=r, wr=wr, al=al, be=be)
    print(f"cached {which} n={nr}")


def analyse(which, nr):
    z = np.load(f"cache_{which}_{nr}.npz")
    r, wr, al, be = z["r"], z["wr"], z["al"], z["be"]
    mom = lambda g, m: float(np.sum(wr * r ** m * g))
    a4, b4 = mom(al, 4), mom(be, 4)
    S6 = b4 / a4
    th = be - S6 * al
    sg = np.sign(th)
    idx = [i for i in range(len(r) - 1) if sg[i] != sg[i + 1]]
    print(f"\n--- {which}  n = {nr} ---")
    print(f"  S(6) = {S6:.8f}   sign changes = {len(idx)}")
    for i in idx:
        print(f"    root in ({r[i]:.6f}, {r[i+1]:.6f})   "
              f"Theta {th[i]:+.6f} -> {th[i+1]:+.6f}")
    # distance from zero away from the roots
    guard = 0.02
    roots = [0.5 * (r[i] + r[i + 1]) for i in idx]
    far = np.ones_like(r, dtype=bool)
    for rt in roots:
        far &= np.abs(r - rt) > guard
    print(f"  |Theta| away from the roots (guard {guard}): "
          f"min {np.min(np.abs(th[far])):.4f}, max {np.max(np.abs(th)):.4f}")
    # transversality: sign of the slope through each bracket
    for i in idx:
        j0, j1 = max(i - 3, 0), min(i + 4, len(r) - 1)
        sl = np.diff(th[j0:j1 + 1]) / np.diff(r[j0:j1 + 1])
        print(f"    slope near root {0.5*(r[i]+r[i+1]):.4f}: "
              f"{sl.min():+.3f} .. {sl.max():+.3f}  "
              f"({'monotone' if np.all(sl > 0) or np.all(sl < 0) else 'NOT MONOTONE'})")
    # the pencil
    lams = np.linspace(S6, 12.5, 60) if which == "srt" else np.linspace(6.5, S6, 60)
    counts = set()
    for lam in lams:
        t = be - lam * al
        s = np.sign(t)
        counts.add(int(np.sum(s[:-1] != s[1:])))
    print(f"  pencil beta - lambda alpha over lambda in "
          f"[{lams[0]:.4f}, {lams[-1]:.4f}]: sign counts observed {sorted(counts)}")
    return S6, len(idx)


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cache(sys.argv[2], int(sys.argv[3]))
    else:
        analyse(sys.argv[2], int(sys.argv[3]))
