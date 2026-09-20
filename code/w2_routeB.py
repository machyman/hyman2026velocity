"""
W2 / ENH-1, experiment 3: review of Route B, and a candidate replacement.

TARGET, restated exactly.  With dC/dD the sigma-averaged deviations,
  S_srt > S_ext   <=>   dC_srt*dD_ext - dC_ext*dD_srt > 0,
the very determinant whose non-vanishing is hypothesis (10).  We want its SIGN.

ROUTE B AS STATED (coupling on matchings).  det = E_{v,w}[F(v,w)] over
independent sigma draws, with
  F(v,w) = (C_srt(v)-Cbar(v))(D_ext(w)-Dbar(w)) - (D_srt(v)-Dbar(v))(C_ext(w)-Cbar(w)).
Any coupling that reduces to a pointwise claim must in particular survive the
diagonal w = v, where F(v,v) is exactly the pointwise determinant.  Experiment 2
showed that is negative for ~16% of states at N=6.  So the diagonal already
refutes the pointwise form of Route B.  Test 1 confirms this directly and asks
whether the pointwise determinant is negative at LARGE N too, which would close
the two-part argument as well.

CANDIDATE REPLACEMENT (Route C, threshold rearrangement).  Put
  w_lambda(i,j) = x_i x_j (x_i + x_j - lambda).
A uniformly random perfect matching gives each pair probability 1/(N-1), so
  E_matching[ sum_P w ] = (1/(N-1)) sum_{i<j} w,
which is exactly the exchangeable baseline.  Hence
  Phi_lambda(rule) := dD_rule - lambda*dC_rule
                    = E_sigma[ sum_{P_rule} w_lambda - avg over matchings ].
Phi is zero at lambda = S_rule.  So if, at lambda = S_ext, the SORTED matching
maximises sum_P w_lambda over all perfect matchings pointwise (strictly with
positive probability), then Phi_{S_ext}(srt) > 0, which is exactly S_srt > S_ext.
Note d^2 w/dx_i dx_j = 2(x_i + x_j) - lambda, so w is supermodular only where
x_i + x_j > lambda/2; global rearrangement does NOT apply and the claim has to
be tested.  Test 2 brute-forces it over every perfect matching at N=6 and N=8.
"""
import numpy as np, itertools
from w2_pointwise import stats

def matchings(idx):
    if not idx: yield []; return
    a, rest = idx[0], idx[1:]
    for k in range(len(rest)):
        b = rest[k]
        for m in matchings(rest[:k]+rest[k+1:]): yield [(a,b)]+m

rng = np.random.default_rng(2718281)

print("TEST 1  pointwise determinant  dC_srt*dD_ext - dC_ext*dD_srt  at large N")
print(f"{'N':>6} {'frac > 0':>10} {'min':>12} {'1st pct':>12}")
for N in [16, 32, 64, 128]:
    M = 300000
    g = rng.standard_normal((M, N))
    v = g/np.linalg.norm(g, axis=1, keepdims=True)*np.sqrt(N)
    Cs, Ce, Cb, Ds, De, Db = stats(v**2)
    det = (Cs-Cb)*(De-Db) - (Ce-Cb)*(Ds-Db)
    print(f"{N:>6} {(det>0).mean():>10.5f} {det.min():>12.4f} {np.percentile(det,1):>12.4f}")

print()
print("TEST 2  does the SORTED matching maximise sum_P w_lambda at lambda = S_ext?")
print("        brute force over every perfect matching; NC: at lambda=0 sorted must win (C is supermodular)")
for N, lam in [(6, 4.534332), (8, 4.884917)]:
    allm = list(matchings(tuple(range(N))))
    M = 20000
    g = rng.standard_normal((M, N)); v = g/np.linalg.norm(g,axis=1,keepdims=True)*np.sqrt(N)
    x = np.sort(v**2, axis=1)
    srt = [(2*k, 2*k+1) for k in range(N//2)]
    def val(pairs, lam):
        return sum(x[:,a]*x[:,b]*(x[:,a]+x[:,b]-lam) for a,b in pairs)
    for L, tag in [(0.0, "NC lambda=0"), (lam, f"lambda=S_ext={lam:.4f}")]:
        best = np.full(M, -np.inf)
        for m in allm: best = np.maximum(best, val(m, L))
        vs = val(srt, L)
        wins = (vs >= best - 1e-10).mean()
        gap = (best - vs).max()
        print(f"  N={N:<3} {tag:<26} sorted optimal in {wins:>8.5f} of states, worst shortfall {gap:>9.4f}")
