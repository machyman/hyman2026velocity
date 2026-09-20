"""
W2c step 1: the PAIRING-PROBABILITY KERNEL reduction.

Claim.  Let y_1 <= ... <= y_N be the order statistics of N iid chi^2_1
variables, u = F(x), v = F(x') uniform coordinates, Q = F^{-1} the chi^2_1
quantile.  For ANY rank-based perfect matching rule, summing a symmetric g
over the rule's pairs is a SINGLE two-dimensional integral:

    sum_{(i,j) in rule} E[g(y_i,y_j)]
        = N(N-1) * int int_{0<u<v<1} g(Q(u),Q(v)) P_rule(u,v) du dv

where P_rule(u,v) is the probability that two points landing at u<v are
matched to each other by the rule, given the other N-2 points are iid uniform.

Sorted rule, pairs (2k+1, 2k+2):  matched iff no other point lies between them
AND an even number lie below.  With a = u, b = 1-v,

    P_srt = (1/2) [ (a+b)^{N-2} + (b-a)^{N-2} ]        (N even, so N-2 even)

Extremal rule, pairs (r, N+1-r):  matched iff #below = #above.  With
c = v-u = 1-a-b,

    P_ext = sum_m  (N-2)! / (m! m! (N-2-2m)!)  (ab)^m c^{N-2-2m}

This replaces N/2 separate singular order-statistic integrals by one smooth
integral per rule, which is what makes an error statement defensible.

NEGATIVE CONTROLS
  K1  N(N-1) int int P_rule = N/2          (the rule has N/2 pairs)
  K2  A, B agree with the order-statistic route of w2_exact.py
  K3  the all-pairs kernel P = 1 reproduces sum M = N(N-1)/2, sum T = 3N(N-1)
"""
import numpy as np
from scipy.special import erfinv, gammaln
from scipy.integrate import quad

Q = lambda p: 2.0 * erfinv(p) ** 2


def P_srt(a, b, N):
    return 0.5 * ((a + b) ** (N - 2) + (b - a) ** (N - 2))


def P_ext(a, b, N):
    c = 1.0 - a - b
    if c < 0:
        return 0.0
    tot = 0.0
    ab = a * b
    for m in range(0, (N - 2) // 2 + 1):
        lg = gammaln(N - 1) - 2 * gammaln(m + 1) - gammaln(N - 1 - 2 * m)
        if ab <= 0.0:
            term = np.exp(lg) * (0.0 ** m) * c ** (N - 2 - 2 * m)
        else:
            term = np.exp(lg + m * np.log(ab)) * c ** (N - 2 - 2 * m)
        tot += term
    return tot


def integrate(g, kern, N, tol=1e-11):
    """N(N-1) * int_{a>0} int_{b>0, a+b<1} g(Q(a), Q(1-b)) kern(a,b,N) db da."""
    def outer(a):
        def inner(b):
            k = kern(a, b, N)
            if k == 0.0:
                return 0.0
            return g(Q(a), Q(1.0 - b)) * k
        val, _ = quad(inner, 0.0, 1.0 - a, epsabs=tol, epsrel=tol, limit=300)
        return val
    val, err = quad(outer, 0.0, 1.0, epsabs=tol, epsrel=tol, limit=300)
    return N * (N - 1) * val, N * (N - 1) * err


gA = lambda x, y: x * y - 1.0
gB = lambda x, y: x * x * y + x * y * y - 6.0
g1 = lambda x, y: 1.0


def arms(N, tol=1e-11):
    out = {}
    for name, kern in (("srt", P_srt), ("ext", P_ext)):
        K1, _ = integrate(g1, kern, N, tol)
        A, eA = integrate(gA, kern, N, tol)
        B, eB = integrate(gB, kern, N, tol)
        out[name] = dict(K1=K1, A=A, B=B, eA=eA, eB=eB)
    return out


if __name__ == "__main__":
    print(f"{'N':>4} {'rule':>4} {'K1 (=N/2)':>12} {'A':>14} {'B':>16} {'S=B/A':>11}")
    for N in [6, 8, 10, 12]:
        r = arms(N)
        for nm in ("srt", "ext"):
            d = r[nm]
            print(f"{N:>4} {nm:>4} {d['K1']:>12.8f} {d['A']:>14.8f} "
                  f"{d['B']:>16.8f} {d['B']/d['A']:>11.6f}")
        det = r["srt"]["A"] * r["ext"]["B"] - r["ext"]["A"] * r["srt"]["B"]
        print(f"{'':>4} {'det':>4} {det:>+12.6f}   "
              f"R = {(r['ext']['A']*r['srt']['B'])/(r['srt']['A']*r['ext']['B']):.6f}")
