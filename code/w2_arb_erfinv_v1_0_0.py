"""Rigorous enclosure of erfinv on arb, built the safe way.

DESIGN: approximate search, RIGOROUS VERIFICATION.  The search may use any
floating-point method; correctness rests only on the final check.

THE FAILURE THIS AVOIDS (Session 42): a bisection that tested `ball < ball`
and treated False as "not less".  For arb, `a < b` returns True only when the
comparison is CERTAIN; False conflates "certainly not less" with "unknown".
A bisection driven by that walks the wrong way on overlap and converges
confidently to a wrong point with zero width.

RULE OBSERVED HERE: ball comparison is used ONLY where True constitutes proof.
False is never read as evidence of anything; it only means "widen and retry".
"""
from flint import arb, ctx
import scipy.special as sp

def erfinv_enclose(p_lo, p_hi, prec=200, pad0=1e-13, max_doublings=80):
    """Return floats (x_lo, x_hi) with erfinv([p_lo,p_hi]) subset [x_lo,x_hi].

    Proof obligation, discharged by arb:
        erf(x_lo) <  p_lo   =>  x_lo <  erfinv(p_lo)
        erf(x_hi) >  p_hi   =>  x_hi >  erfinv(p_hi)
    erf is strictly increasing, so these give the containment.
    """
    assert 0.0 < p_lo <= p_hi < 1.0, (p_lo, p_hi)
    ctx.prec = prec
    xl = float(sp.erfinv(p_lo)); xh = float(sp.erfinv(p_hi))
    Plo, Phi = arb(p_lo), arb(p_hi)
    pad = pad0
    for _ in range(max_doublings):
        a = xl - pad; b = xh + pad
        if arb(a).erf() < Plo and arb(b).erf() > Phi:   # True == proved
            return a, b
        pad *= 2.0
    raise RuntimeError("erfinv_enclose: could not verify for [%r,%r]" % (p_lo, p_hi))

def Q_enclose(p_lo, p_hi, prec=200):
    """Enclose Q(p) = 2*erfinv(p)^2 over p in [p_lo,p_hi].  Q is increasing on (0,1)."""
    a, b = erfinv_enclose(p_lo, p_hi, prec)
    return 2.0*a*a, 2.0*b*b        # a >= 0 on (0,1), so squaring is monotone

def R_enclose(b_lo, b_hi, prec=200):
    """Enclose R(b) = 2*erfcinv(b)^2 = 2*erfinv(1-b)^2, DECREASING in b."""
    lo, hi = erfinv_enclose(1.0-b_hi, 1.0-b_lo, prec)
    return 2.0*lo*lo, 2.0*hi*hi
