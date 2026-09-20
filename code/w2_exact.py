"""
W2 / ENH-1, Route D: EXACT evaluation, replacing Monte Carlo.

By the homogeneity reduction (w2_routeD.py) the sphere drops out and everything
is an expectation over N INDEPENDENT chi^2_1 variables.  With y the order
statistics and u ~ chi^2_1 (E[u]=1, E[u^2]=3),

    E[Cbar] = N/2 ,   E[Dbar] = 3N          (exact, no quadrature needed)
    A_rule  = sum over the rule's pairs of M_ij  -  N/2
    B_rule  = sum over the rule's pairs of T_ij  -  3N
    M_ij = E[y_i y_j] ,  T_ij = E[y_i^2 y_j] + E[y_i y_j^2]

and the theorem at this N is  det = A_srt*B_ext - A_ext*B_srt > 0.

Each M, T is a TWO-dimensional integral against the explicit bivariate
order-statistic density.  In uniform coordinates 0<p<q<1 that density is
    c_ij * p^(i-1) * (q-p)^(j-i-1) * (1-q)^(N-j),
    c_ij = N!/((i-1)!(j-i-1)!(N-j)!),
and y = Q(p) with Q(p) = 2*erfinv(p)^2, the chi^2_1 quantile.

NEGATIVE CONTROLS, all checked below:
  NC1  sum_i E[y_i] = N          (order statistics partition the total)
  NC2  sum_{i<j} M_ij = N(N-1)/2 (= E[sum_{i<j} u_i u_j], E[u]=1)
  NC3  sum_{i<j} T_ij = 3N(N-1)  (= E[sum_{i!=j} u_i^2 u_j], E[u^2]E[u]=3)
  NC4  the resulting R(N) must reproduce the Monte Carlo value
"""
import numpy as np
from scipy.special import erfinv, gammaln
from scipy.integrate import quad

Q = lambda p: 2.0*erfinv(p)**2

def logc(N,i,j):
    return (gammaln(N+1) - gammaln(i) - gammaln(j-i) - gammaln(N-j+1))

def moment(N, i, j, a, b, tol=1e-12):
    """E[ y_i^a y_j^b ] for i<j, order statistics of N iid chi^2_1."""
    C = np.exp(logc(N,i,j))
    def outer(p):
        qa = Q(p)**a
        if qa == 0.0: return 0.0
        def inner(r):
            q = p + (1.0-p)*r
            if q >= 1.0: return 0.0
            return (q-p)**(j-i-1) * (1.0-q)**(N-j) * Q(q)**b
        val, _ = quad(inner, 0.0, 1.0, epsabs=tol, epsrel=tol, limit=200)
        return qa * p**(i-1) * (1.0-p) * val
    val, err = quad(outer, 0.0, 1.0, epsabs=tol, epsrel=tol, limit=200)
    return C*val, C*err

def uni_moment(N, i, a, tol=1e-12):
    C = np.exp(gammaln(N+1)-gammaln(i)-gammaln(N-i+1))
    f = lambda p: Q(p)**a * p**(i-1) * (1.0-p)**(N-i)
    val, err = quad(f, 0.0, 1.0, epsabs=tol, epsrel=tol, limit=200)
    return C*val, C*err

for N in [6, 8]:
    print(f"===== N = {N} =====")
    s1 = sum(uni_moment(N,i,1)[0] for i in range(1,N+1))
    print(f"  NC1  sum E[y_i]      = {s1:.10f}   exact {N}      err {abs(s1-N):.2e}")
    M = {}; T = {}
    for i in range(1,N+1):
        for j in range(i+1,N+1):
            M[(i,j)] = moment(N,i,j,1,1)[0]
            T[(i,j)] = moment(N,i,j,2,1)[0] + moment(N,i,j,1,2)[0]
    sM, sT = sum(M.values()), sum(T.values())
    print(f"  NC2  sum M_ij        = {sM:.10f}   exact {N*(N-1)/2}   err {abs(sM-N*(N-1)/2):.2e}")
    print(f"  NC3  sum T_ij        = {sT:.10f}   exact {3*N*(N-1)}   err {abs(sT-3*N*(N-1)):.2e}")
    srt = [(2*k+1, 2*k+2) for k in range(N//2)]
    ext = [(k+1, N-k)     for k in range(N//2)]
    A_s = sum(M[p] for p in srt) - N/2;  B_s = sum(T[p] for p in srt) - 3*N
    A_e = sum(M[p] for p in ext) - N/2;  B_e = sum(T[p] for p in ext) - 3*N
    det = A_s*B_e - A_e*B_s
    R   = (B_s/A_s)/(B_e/A_e)
    print(f"  A_srt = {A_s:+.10f}   B_srt = {B_s:+.10f}   S_srt = {B_s/A_s:.8f}")
    print(f"  A_ext = {A_e:+.10f}   B_ext = {B_e:+.10f}   S_ext = {B_e/A_e:.8f}")
    print(f"  determinant  A_srt*B_ext - A_ext*B_srt = {det:+.8f}   -> {'POSITIVE' if det>0 else 'NEGATIVE'}")
    print(f"  NC4  R({N}) = {R:.6f}   Monte Carlo {1.1241 if N==6 else 1.2077}")
