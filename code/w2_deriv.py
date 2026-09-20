"""
W2c-cert, sorted arm: alpha' and beta' in closed form.

Session 28 read the Wronskian W = alpha beta' - alpha' beta off finite
differences of alpha and beta sampled at Gauss nodes.  Non-uniform spacing and
endpoint one-sidedness make that a weak instrument for a sign claim, and the
certification needs alpha' and beta' anyway.  Both come from differentiating
under the integral sign, once the limits are fixed by moving to b.

Write u = 1-b, a for the lower argument, and let F be the integrand core,

    F_A(a,u) = Q(a)Q(u) - 1,        d/da F_A = Q'(a) Q(u)
    F_B(a,u) = Q(a)Q(u)(Q(a)+Q(u)) - 6,
                                    d/da F_B = Q'(a) Q(u) (2Q(a) + Q(u))

With m = (1-r)/2 and M = (1+r)/2:

    Phi(r)      = 2 int_0^r F(r-b, 1-b) db
    Phi'(r)     = 2 F(0, 1-r) + 2 int_0^r dF/da (r-b, 1-b) db

    Psi(r)      = 2 int_r^{M} F(b-r, 1-b) db
    Psi'(r)     = 2[ -F(0,1-r) + (1/2) F(m,m) - int_r^{M} dF/da (b-r,1-b) db ]

    Psi(-r)     = 2 int_0^{m} F(b+r, 1-b) db
    d/dr Psi(-r)= 2[ -(1/2) F(M,M) + int_0^{m} dF/da (b+r,1-b) db ]

so alpha' = Phi'_A + Psi'_A + d/dr Psi_A(-r), and likewise beta'.  Note that
F(0, .) = -1 for A and -6 for B, since Q(0) = 0, so the divergent-looking
boundary term at b -> r is finite; the logarithmic endpoint is at b -> 0,
where Q(1-b) blows up but Q'(a) stays bounded, and is tamed by b = c t^2.

Q'(p) = 2 sqrt(pi) erfinv(p) exp(erfinv(p)^2).
"""
import numpy as np
from scipy.special import erfinv, erfcinv

SQPI = np.sqrt(np.pi)


def QL(a):                      # Q(a), stable for a near 0
    return 2.0 * erfinv(a) ** 2


def QU(b):                      # Q(1-b), stable for b near 0
    return 2.0 * erfcinv(b) ** 2


def dQ(a):                      # Q'(a)
    e = erfinv(a)
    return 2.0 * SQPI * e * np.exp(e * e)


def gl(n, lo=0.0, hi=1.0):
    p, w = np.polynomial.legendre.leggauss(n)
    return lo + (hi - lo) * 0.5 * (p + 1.0), 0.5 * (hi - lo) * w


# integrand cores, given a and b (u = 1-b handled inside)
def FA(a, b):
    return QL(a) * QU(b) - 1.0


def FB(a, b):
    x, y = QL(a), QU(b)
    return x * y * (x + y) - 6.0


def dFA(a, b):
    return dQ(a) * QU(b)


def dFB(a, b):
    x, y = QL(a), QU(b)
    return dQ(a) * y * (2.0 * x + y)


def _sq(n, hi):
    """nodes on (0,hi) with b = hi t^2, for the logarithmic endpoint at 0."""
    t, w = gl(n)
    return hi * t ** 2, w * 2.0 * hi * t


def alpha_beta_and_derivs(r, n=800):
    m, M = 0.5 * (1.0 - r), 0.5 * (1.0 + r)
    out = {}
    for tag, F, dF, zero in (("A", FA, dFA, -1.0), ("B", FB, dFB, -6.0)):
        # Phi(r) = 2 int_0^r F(r-b, b) db ; endpoint b -> 0 is logarithmic
        b, w = _sq(n, r)
        Phi = 2.0 * np.sum(w * F(r - b, b))
        dPhi = 2.0 * zero + 2.0 * np.sum(w * dF(r - b, b))
        # Psi(r) = 2 int_r^M F(b-r, b) db ; no singular endpoint
        b2, w2 = gl(n, r, M)
        Psi = 2.0 * np.sum(w2 * F(b2 - r, b2))
        # at b = M: a = M-r = m and u = 1-M = m, so pass b = 1-m = M
        Fmm = F(m, M)
        dPsi = 2.0 * (-zero + 0.5 * Fmm - np.sum(w2 * dF(b2 - r, b2)))
        # Psi(-r) = 2 int_0^m F(b+r, b) db ; endpoint b -> 0 is logarithmic
        b3, w3 = _sq(n, m)
        Psi_m = 2.0 * np.sum(w3 * F(b3 + r, b3))
        # at b = m: a = m+r = M and u = 1-m = M, so pass b = 1-M = m
        FMM = F(M, m)
        dPsi_m = 2.0 * (-0.5 * FMM + np.sum(w3 * dF(b3 + r, b3)))
        out[tag] = (Phi + Psi + Psi_m, dPhi + dPsi + dPsi_m)
    (al, dal), (be, dbe) = out["A"], out["B"]
    return al, be, dal, dbe


if __name__ == "__main__":
    print("check: F(m,m) uses a = u = m, so Q(a)Q(u) = Q(m)^2 as derived\n")
    print(f"{'r':>7} {'alpha':>12} {'beta':>12} {'alpha-prime':>13} "
          f"{'beta-prime':>13} {'W':>14} {'W fd':>14}")
    rs = [0.001, 0.01, 0.05, 0.2, 0.41, 0.6, 0.8, 0.92, 0.98, 0.999]
    prev = None
    h = 1e-5
    for r in rs:
        al, be, dal, dbe = alpha_beta_and_derivs(r)
        W = al * dbe - dal * be
        a1, b1, _, _ = alpha_beta_and_derivs(min(r + h, 1.0 - 1e-12))
        a0, b0, _, _ = alpha_beta_and_derivs(max(r - h, 1e-12))
        Wfd = al * (b1 - b0) / (2 * h) - (a1 - a0) / (2 * h) * be
        print(f"{r:>7.3f} {al:>12.6f} {be:>12.6f} {dal:>13.6f} {dbe:>13.6f} "
              f"{W:>14.6f} {Wfd:>14.6f}")
