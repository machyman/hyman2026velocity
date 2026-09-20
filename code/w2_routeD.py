"""
W2 / ENH-1, experiment 4: Route D — remove the sphere constraint entirely.

C and Cbar are homogeneous of degree 2 in x, D and Dbar of degree 3.  On the
sphere x = N*u/sum(u) with u_i = g_i^2 iid chi^2_1, and u/sum(u) is INDEPENDENT
of sum(u).  So for f homogeneous of degree d,
        E_sigma[f(x)] = N^d * E[f(u)] / E[(sum u)^d],   sum u ~ chi^2_N.
The factor is common to both rules, so with
        A_rule = E[ C_rule(u) - Cbar(u) ],   B_rule = E[ D_rule(u) - Dbar(u) ]
computed for u_1..u_N INDEPENDENT chi^2_1, every normalisation cancels in the
ratio and
        R(N) = S_srt/S_ext = (B_srt/A_srt) / (B_ext/A_ext).
The determinant keeps its sign: det > 0  <=>  A_srt*B_ext - A_ext*B_srt > 0.

If this reproduces R(N), the theorem is an inequality about order statistics of
N INDEPENDENT variables, not about a Dirichlet on the sphere, and each A,B is a
sum of O(N^2) TWO-dimensional integrals against the explicit bivariate
order-statistic density.  That makes every fixed N a finite rigorous
computation rather than an N-dimensional one.
"""
import numpy as np
rng = np.random.default_rng(16180339)
print(f"{'N':>6} {'R via iid chi2_1':>18} {'R on the sphere':>17} {'manuscript':>11}")
ref = {6:1.1241, 8:1.2077, 16:1.3846, 64:1.6207, 256:1.7341}
for N in [6, 8, 16, 64, 256]:
    M = 4000000
    y = np.sort(rng.chisquare(1.0, size=(M, N)), axis=1)
    a_s, b_s = y[:, 0::2], y[:, 1::2]
    a_e, b_e = y[:, :N//2], y[:, N//2:][:, ::-1]
    T1 = y.sum(1); T2 = (y**2).sum(1); T3 = (y**3).sum(1)
    Cbar = (T1**2 - T2) / (2*(N-1))
    Dbar = (T1*T2 - T3) / (N-1)
    A_s = ((a_s*b_s).sum(1) - Cbar).mean()
    A_e = ((a_e*b_e).sum(1) - Cbar).mean()
    B_s = ((a_s*b_s*(a_s+b_s)).sum(1) - Dbar).mean()
    B_e = ((a_e*b_e*(a_e+b_e)).sum(1) - Dbar).mean()
    R_iid = (B_s/A_s)/(B_e/A_e)
    # sphere, same sample, for a like-for-like check
    x = N*y/T1[:, None]
    xa_s, xb_s = x[:, 0::2], x[:, 1::2]
    xa_e, xb_e = x[:, :N//2], x[:, N//2:][:, ::-1]
    S2 = (x**2).sum(1); S3 = (x**3).sum(1)
    cb = (N**2 - S2)/(2*(N-1)); db = (N*S2 - S3)/(N-1)
    dCs = ((xa_s*xb_s).sum(1)-cb).mean(); dDs = ((xa_s*xb_s*(xa_s+xb_s)).sum(1)-db).mean()
    dCe = ((xa_e*xb_e).sum(1)-cb).mean(); dDe = ((xa_e*xb_e*(xa_e+xb_e)).sum(1)-db).mean()
    R_sph = (dDs/dCs)/(dDe/dCe)
    print(f"{N:>6} {R_iid:>18.5f} {R_sph:>17.5f} {ref[N]:>11.4f}")
