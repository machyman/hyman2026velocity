"""
W2 / ENH-1, experiment 2: with the uniform separating constant ruled out,
does the WEAKER pointwise statement S_srt(v) > S_ext(v) hold state by state?

If yes, a pointwise inequality survives and the remaining gap is only the
(standard, but real) step from a pointwise ratio inequality to an inequality
between ratios of expectations.
If no, no pointwise route survives at all and the sigma-average is doing
essential work, which changes what has to be proved.
"""
import numpy as np
from w2_pointwise import stats
rng = np.random.default_rng(31415926)
print(f"{'N':>5} {'frac S_srt>S_ext':>17} {'min(S_srt-S_ext)':>18} {'5th pct':>10} {'median':>10}")
for N in [4, 6, 8, 10, 16, 32, 64]:
    M = 400000
    g = rng.standard_normal((M, N))
    v = g/np.linalg.norm(g, axis=1, keepdims=True)*np.sqrt(N)
    C_srt, C_ext, Cbar, D_srt, D_ext, Dbar = stats(v**2)
    dCs, dDs = C_srt-Cbar, D_srt-Dbar
    dCe, dDe = C_ext-Cbar, D_ext-Dbar
    ok = (dCs > 1e-12) & (dCe < -1e-12)
    diff = dDs[ok]/dCs[ok] - dDe[ok]/dCe[ok]
    print(f"{N:>5} {(diff>0).mean():>17.4f} {diff.min():>18.4f} "
          f"{np.percentile(diff,5):>10.4f} {np.median(diff):>10.4f}")
