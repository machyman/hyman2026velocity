"""
W2c step 4 (rebuilt): evaluate alpha and beta ONCE on a Gauss-Legendre grid in
r, then take every moment as a weighted sum.  The previous design called the
adaptive integrator per moment, which re-evaluated a 10^5-point double
integral hundreds of times per moment.

Controls carried in this script:
  C-a   a_0 = b_0 = 0 for both rules              (the N = 2 degeneracy)
  C-b   A(N), B(N) reconstructed from the grid moments must reproduce the
        values established by the order-statistic route and the kernel route
  C-c   the grid is refined and the answers must not move
"""
import sys
import numpy as np
from scipy.special import erfinv, erfcinv

# ---------------------------------------------------------------- integrands
def fA_vec(a, b):
    out = np.full(np.broadcast(a, b).shape, -1.0)
    ok = (a > 0) & (b > 0)
    out[ok] = 2.0 * erfinv(a[ok]) ** 2 * 2.0 * erfcinv(b[ok]) ** 2 - 1.0
    return out


def fB_vec(a, b):
    out = np.full(np.broadcast(a, b).shape, -6.0)
    ok = (a > 0) & (b > 0)
    x = 2.0 * erfinv(a[ok]) ** 2
    y = 2.0 * erfcinv(b[ok]) ** 2
    out[ok] = x * y * (x + y) - 6.0
    return out


def gl(n, lo=0.0, hi=1.0):
    p, w = np.polynomial.legendre.leggauss(n)
    return lo + (hi - lo) * 0.5 * (p + 1.0), 0.5 * (hi - lo) * w


# ------------------------------------------------------------- sorted arm
def F_srt(fv, r, n=800):
    """Phi(r) + Psi(r) + Psi(-r), all written in b with b = (endpoint) t^2
    where the b -> 0 endpoint carries a logarithmic singularity."""
    t, w = gl(n)
    # Phi(r) = int_{-r}^{r} f dd = 2 int_0^r f(a=r-b, b) db
    b = r * t ** 2
    tot = np.sum(w * (4.0 * r * t) * fv(r - b, b))
    # Psi(r) = int_r^1 f(s, d=r) ds ; b = (s+r)/2 from r to (1+r)/2, no edge
    b0, b1 = r, 0.5 * (1.0 + r)
    bb = b0 + (b1 - b0) * t
    tot += np.sum(w * (2.0 * (b1 - b0)) * fv(bb - r, bb))
    # Psi(-r) = int_r^1 f(s, d=-r) ds ; b = (s-r)/2 from 0 to (1-r)/2, edge at 0
    b1 = 0.5 * (1.0 - r)
    bb = b1 * t ** 2
    tot += np.sum(w * (4.0 * b1 * t) * fv(bb + r, bb))
    return float(tot)


# ------------------------------------------------------------ extremal arm
def F_ext(fv, r, ns=300, nphi=300):
    tau, wt = gl(ns)
    phi, wp = gl(nphi, -0.5 * np.pi, 0.5 * np.pi)
    wp = wp / np.pi
    tot = 0.0
    for x in (r, -r):
        lo = (1.0 - r) / 2.0 if x > 0 else (1.0 + r) / 2.0
        if lo >= 1.0:
            continue
        s = lo + (1.0 - lo) * tau ** 2
        jac = 2.0 * (1.0 - lo) * tau
        e = x - 1.0 + s
        dm = np.sqrt(np.maximum(s * s - e * e, 0.0))
        d = dm[:, None] * np.sin(phi)[None, :]
        S = np.broadcast_to(s[:, None], d.shape)
        vals = fv(0.5 * (S - d), 0.5 * (S + d))
        tot += np.sum(wt * jac * (vals @ wp))
    return float(0.5 * tot)


# ------------------------------------------------------------------ driver
def run(label, Ffun, nr, pre, ref):
    r, wr = gl(nr)
    al = np.array([Ffun(fA_vec, x) for x in r])
    be = np.array([Ffun(fB_vec, x) for x in r])
    mom = lambda g, m: float(np.sum(wr * r ** m * g))
    a0, b0 = mom(al, 0), mom(be, 0)
    a4, b4 = mom(al, 4), mom(be, 4)
    S6 = b4 / a4
    print(f"\n=== {label}   (r-grid n = {nr}) ===")
    print(f"  C-a  a_0 = {a0:+.3e}   b_0 = {b0:+.3e}      (both must vanish)")
    print(f"  C-b  reconstruction from grid moments:")
    print(f"       {'N':>4} {'A':>14} {'B':>15} {'S':>11} {'ref S':>11} {'rel':>9}")
    for N, rs in ref.items():
        A = pre(N) * mom(al, N - 2)
        B = pre(N) * mom(be, N - 2)
        print(f"       {N:>4} {A:>14.8f} {B:>15.8f} {B/A:>11.6f} {rs:>11.6f} "
              f"{abs(B/A - rs)/rs:>9.1e}")
    th = be - S6 * al
    sg = np.sign(th)
    idx = [i for i in range(len(r) - 1) if sg[i] != sg[i + 1]]
    print(f"  S(6) from the grid = {S6:.6f}")
    print(f"  SIGN CHANGES of Theta = beta - S(6) alpha : {len(idx)}")
    for i in idx:
        print(f"       root bracketed in ({r[i]:.5f}, {r[i+1]:.5f})")
    first = "+" if th[0] > 0 else "-"
    last = "+" if th[-1] > 0 else "-"
    mid = "+" if th[len(th) // 2] > 0 else "-"
    print(f"  pattern at (r->0, mid, r->1): ({first}, {mid}, {last})")
    print(f"  moments int r^m Theta:")
    for m in [0, 4, 6, 8, 12, 20, 40, 80]:
        print(f"       m = {m:>3}   {mom(th, m):+.9e}")
    return S6, len(idx)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "srt"
    nr = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if which == "srt":
        run("SORTED", F_srt, nr, lambda N: N * (N - 1) / 4.0,
            {6: 8.494413, 8: 8.850005, 12: 9.339742, 24: 10.097097})
    else:
        run("EXTREMAL", F_ext, nr, lambda N: N * (N - 1) * 1.0,
            {6: 7.555817, 8: 7.328171, 12: 7.099443, 24: 6.872340})
