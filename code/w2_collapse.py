"""
W2c step 2: COLLAPSE TO ONE DIMENSION.

The sorted kernel is a mixture of pure powers.  With a = u, b = 1-v,
s = a+b, d = b-a (so a = (s-d)/2, b = (s+d)/2, region |d| < s < 1),

    P_srt = (1/2)[ s^{N-2} + d^{N-2} ]

and da db = (1/2) ds dd, so for any f(a,b),

    sum_{sorted pairs} E[f] = N(N-1) int int f P_srt
                            = (N(N-1)/4) int_0^1 r^{N-2} F(r) dr ,
    F(r) = Phi_f(r) + Psi_f(r) + Psi_f(-r) ,
    Phi_f(s) = int_{-s}^{s} f(s,d) dd ,   Psi_f(d) = int_{|d|}^{1} f(s,d) ds .

(The d-integral is folded using that N-2 is even.)  The N-dependence is now a
single power r^{N-2} against an N-INDEPENDENT function.  Writing

    alpha = F for f = QL*QU - 1 ,       beta = F for f = QL*QU*(QL+QU) - 6 ,
    a_m = int_0^1 r^m alpha ,           b_m = int_0^1 r^m beta ,

we have S_srt(N) = b_{N-2} / a_{N-2}, and

    b_{m+1} a_m - b_m a_{m+1}
        = (1/2) int int (rr')^m (r-r') [alpha(r')beta(r) - alpha(r)beta(r')] ,

so S_srt is nondecreasing in N as soon as the 2-by-n matrix (alpha; beta) is
totally positive of order 2 on (0,1):

    D(r,r') = alpha(r') beta(r) - alpha(r) beta(r') >= 0   whenever r > r'.

No positivity of alpha is needed.  This script builds alpha and beta and
checks the collapse against the values already established by two independent
routes.
"""
import numpy as np
from scipy.special import erfinv, erfcinv
from scipy.integrate import quad

QL = lambda a: 2.0 * erfinv(a) ** 2          # Q(a),   a -> 0 gives 0
QU = lambda b: 2.0 * erfcinv(b) ** 2         # Q(1-b), b -> 0 blows up as log


def fA(s, d):
    a = 0.5 * (s - d)
    b = 0.5 * (s + d)
    if a <= 0.0 or b <= 0.0:
        return -1.0
    return QL(a) * QU(b) - 1.0


def fB(s, d):
    a = 0.5 * (s - d)
    b = 0.5 * (s + d)
    if a <= 0.0 or b <= 0.0:
        return -6.0
    x, y = QL(a), QU(b)
    return x * y * (x + y) - 6.0


def Phi(f, s, tol=1e-11):
    val, _ = quad(lambda d: f(s, d), -s, s, epsabs=tol, epsrel=tol, limit=400)
    return val


def Psi(f, d, tol=1e-11):
    lo = abs(d)
    if lo >= 1.0:
        return 0.0
    val, _ = quad(lambda s: f(s, d), lo, 1.0, epsabs=tol, epsrel=tol, limit=400)
    return val


def F(f, r, tol=1e-11):
    return Phi(f, r, tol) + Psi(f, r, tol) + Psi(f, -r, tol)


alpha = lambda r: F(fA, r)
beta = lambda r: F(fB, r)


def moment(g, m, tol=1e-10):
    val, err = quad(lambda r: r ** m * g(r), 0.0, 1.0,
                    epsabs=tol, epsrel=tol, limit=400)
    return val, err


def arms(N):
    """A_srt, B_srt via the one-dimensional collapse."""
    pre = N * (N - 1) / 4.0
    a_m, _ = moment(alpha, N - 2)
    b_m, _ = moment(beta, N - 2)
    return pre * a_m, pre * b_m


if __name__ == "__main__":
    import warnings
    ref = {6: (3.19719807, 27.15832093, 8.494413),
           8: (5.02319765, 44.45532477, 8.850005),
           10: (6.89964184, 62.94062376, 9.122303),
           12: (8.80591941, 82.24501189, 9.339742),
           16: (None, None, 9.669320),
           24: (None, None, 10.097097)}
    print(f"{'N':>4} {'A_srt':>14} {'B_srt':>16} {'S_srt':>11} {'ref S':>11} {'rel err':>10}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for N in sorted(ref):
            A, B = arms(N)
            S = B / A
            rs = ref[N][2]
            print(f"{N:>4} {A:>14.8f} {B:>16.8f} {S:>11.6f} {rs:>11.6f} "
                  f"{abs(S - rs) / rs:>10.1e}")
        print(f"\nIntegrationWarnings raised (not suppressed): {len(w)}")
