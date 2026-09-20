"""
W2c step 3: the EXTREMAL rule collapses the same way.

Two points at u < v are matched by the extremal rule iff, among the other N-2
points, as many fall below u as above v.  With a = u, b = 1-v, c = 1-a-b,

    P_ext = sum_m (N-2)!/(m! m! (N-2-2m)!) (ab)^m c^{N-2-2m}.

Using the central binomial coefficient's Fourier form, C(2m,m) = the constant
term of (2 cos t)^{2m},

    P_ext = (1/2pi) int_{-pi}^{pi} rho^{N-2} dt ,
    rho(t) = c + 2 sqrt(ab) cos t = (1-s) + sqrt(s^2-d^2) cos t ,

which is REAL and lies in [-1,1].  So the extremal kernel, like the sorted
one, is a mixture of pure powers r^{N-2} with an N-independent mixing measure,
and the same one-dimensional collapse applies.  Pushing forward the uniform t
through rho gives an arcsine law on (c-R, c+R), R = sqrt(s^2-d^2); folding
|rho| (N-2 is even) and substituting d = d_max sin(phi) cancels the arcsine
singularity exactly, leaving a smooth double integral:

    alpha_ext(r) = (1/2) sum_{x in {r,-r}} int ds (1/pi) int_{-pi/2}^{pi/2}
                       f(s, d_max(s,x) sin phi) dphi ,
    d_max^2 = s^2 - (x-1+s)^2 ,

the s-range being s > (1-r)/2 for x = r and s > (1+r)/2 for x = -r.  Then

    A_ext(N) = N(N-1) int_0^1 r^{N-2} alpha_ext(r) dr,   likewise B_ext.
"""
import numpy as np
from scipy.integrate import quad
from w2_collapse import fA, fB

NPHI = 400
_ph, _wh = np.polynomial.legendre.leggauss(NPHI)
PHI = 0.5 * np.pi * _ph          # nodes on (-pi/2, pi/2)
WPHI = 0.5 * np.pi * _wh / np.pi  # weights, already divided by pi


def _inner(f, s, x):
    e = x - 1.0 + s
    dm2 = s * s - e * e
    if dm2 <= 0.0:
        return 0.0
    dm = np.sqrt(dm2)
    tot = 0.0
    for p, w in zip(PHI, WPHI):
        tot += w * f(s, dm * np.sin(p))
    return tot


def _one(f, r, x, tol=1e-10):
    lo = (1.0 - r) / 2.0 if x > 0 else (1.0 + r) / 2.0
    if lo >= 1.0:
        return 0.0
    val, _ = quad(lambda s: _inner(f, s, x), lo, 1.0,
                  epsabs=tol, epsrel=tol, limit=200)
    return val


def Fext(f, r):
    return 0.5 * (_one(f, r, r) + _one(f, r, -r))


alpha_ext = lambda r: Fext(fA, r)
beta_ext = lambda r: Fext(fB, r)


def moment(g, m, tol=1e-9):
    val, err = quad(lambda r: r ** m * g(r), 0.0, 1.0,
                    epsabs=tol, epsrel=tol, limit=200)
    return val, err


def arms(N):
    a, _ = moment(alpha_ext, N - 2)
    b, _ = moment(beta_ext, N - 2)
    return N * (N - 1) * a, N * (N - 1) * b


if __name__ == "__main__":
    import warnings
    ref = {6: (-1.89244690, -14.29898240, 7.555817),
           8: (-2.80369174, -20.54593247, 7.328171),
           10: (-3.70427079, -26.63705656, 7.190904),
           12: (-4.59861408, -32.64759772, 7.099443)}
    print(f"{'N':>4} {'A_ext':>14} {'ref A':>14} {'B_ext':>15} {'ref B':>15} {'S_ext':>10} {'rel':>9}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for N in sorted(ref):
            A, B = arms(N)
            rA, rB, rS = ref[N]
            print(f"{N:>4} {A:>14.8f} {rA:>14.8f} {B:>15.8f} {rB:>15.8f} "
                  f"{B/A:>10.6f} {abs(B/A - rS)/rS:>9.1e}")
        a0, _ = moment(alpha_ext, 0)
        b0, _ = moment(beta_ext, 0)
        print(f"\nN=2 degeneracy control:  int alpha_ext = {a0:+.3e}   "
              f"int beta_ext = {b0:+.3e}   (both must vanish)")
        print(f"IntegrationWarnings raised (not suppressed): {len(w)}")
