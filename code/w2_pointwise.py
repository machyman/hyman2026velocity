"""
W2 / ENH-1, experiment 1: does a POINTWISE separating constant exist?

Target inequality (proposed route):
  find lambda = lambda(N) with, for EVERY state v on the sphere,
      D(P) - Dbar  >=  lambda * ( C(P) - Cbar )      for P = sorted AND P = extremal.
  Since C_srt - Cbar > 0 > C_ext - Cbar, dividing gives
      S_srt(v) >= lambda >= S_ext(v)  pointwise,
  and taking sigma-expectations of the two linear inequalities gives
      delta_D^srt >= lambda delta_C^srt  and  delta_D^ext <= lambda delta_C^ext,
  hence S_srt >= lambda >= S_ext at that N.  No asymptotics, no finite gaps.

Decisive test: is   sup_v S_ext(v)  <  inf_v S_srt(v)  ?
If the two pointwise ranges are DISJOINT, any lambda in the gap works and the
theorem reduces to two pointwise algebraic inequalities.
If they OVERLAP, the pointwise route fails and the sigma-average is essential.

Negative control: at N=4, eq:n4 forces S(v) == N == 4 identically for BOTH rules.
"""
import numpy as np

def stats(x):
    """x: (M,N) squared velocities, rows sum to N. Returns C_srt,C_ext,Cbar,D_srt,D_ext,Dbar."""
    M, N = x.shape
    xs = np.sort(x, axis=1)                    # ascending
    # sorted rule: pair adjacent ranks (1,2)(3,4)...
    a_s, b_s = xs[:, 0::2], xs[:, 1::2]
    # extremal rule: pair (1,N)(2,N-1)...
    a_e, b_e = xs[:, :N//2], xs[:, N//2:][:, ::-1]
    C_srt = (a_s*b_s).sum(1);            D_srt = (a_s*b_s*(a_s+b_s)).sum(1)
    C_ext = (a_e*b_e).sum(1);            D_ext = (a_e*b_e*(a_e+b_e)).sum(1)
    Q  = (x**2).sum(1)                          # sum x_i^2
    Q6 = (x**3).sum(1)                          # sum x_i^3
    Cbar = (N**2 - Q) / (2*(N-1))               # (1/(N-1)) sum_{i<j} x_i x_j
    Dbar = (N*Q - Q6) / (N-1)
    return C_srt, C_ext, Cbar, D_srt, D_ext, Dbar

rng = np.random.default_rng(20260825)
print(f"{'N':>6} {'inf S_srt(v)':>13} {'sup S_ext(v)':>13} {'gap':>9}  {'verdict':<12} "
      f"{'E-ratio S_srt':>13} {'E-ratio S_ext':>13}")
for N in [4, 6, 8, 10, 16, 32, 64]:
    M = 400000
    g = rng.standard_normal((M, N))
    v = g / np.linalg.norm(g, axis=1, keepdims=True) * np.sqrt(N)
    x = v**2
    C_srt, C_ext, Cbar, D_srt, D_ext, Dbar = stats(x)
    dCs, dDs = C_srt - Cbar, D_srt - Dbar
    dCe, dDe = C_ext - Cbar, D_ext - Dbar
    ok = (dCs > 1e-12) & (dCe < -1e-12)
    Ss, Se = dDs[ok]/dCs[ok], dDe[ok]/dCe[ok]
    lo, hi = Ss.min(), Se.max()
    verdict = "DISJOINT" if lo > hi else "OVERLAP"
    # ratio-of-expectations (the quantity the theorem is actually about)
    Rs, Re = dDs.mean()/dCs.mean(), dDe.mean()/dCe.mean()
    print(f"{N:>6} {lo:>13.6f} {hi:>13.6f} {lo-hi:>9.4f}  {verdict:<12} {Rs:>13.6f} {Re:>13.6f}")
