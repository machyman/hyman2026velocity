"""Rigorous enclosure of W = alpha*beta' - alpha'*beta on the sorted arm.

Monotonicity used (all on (0,1), all factors positive):
    Q(a)  = 2 erfinv(a)^2                 increasing in a
    R(b)  = 2 erfcinv(b)^2                decreasing in b
    dQ(a) = 2 sqrt(pi) e exp(e^2), e=erfinv(a)   increasing in a
so every integrand below is monotone in each variable separately and is
bracketed by its corner values over a box.

Derivative identity, verified against w2_deriv.py to float precision:
    alpha' = 2 I1 + F(m,M) - 2 I2 - F(M,m) + 2 I3
with I1 = int_0^r dF(r-b,b) db, I2 = int_r^M dF(b-r,b) db, I3 = int_0^m dF(b+r,b) db,
m = (1-r)/2, M = (1+r)/2.  The +-2*zero constants cancel.
"""
import math
from flint import arb, ctx
from w2_arb_erfinv_v1_0_0 import erfinv_enclose

EPS = 1e-13
SQPI = math.sqrt(math.pi)

def _Q(a_lo, a_hi):
    e0, e1 = erfinv_enclose(a_lo, a_hi)
    return 2.0*e0*e0, 2.0*e1*e1

def _R(b_lo, b_hi):
    e0, e1 = erfinv_enclose(1.0-b_hi, 1.0-b_lo)
    return 2.0*e0*e0, 2.0*e1*e1

def _dQ(a_lo, a_hi, prec=200):
    """dQ = 2 sqrt(pi) e exp(e^2), increasing.  Evaluated in arb, bounds taken outward."""
    ctx.prec = prec
    e0, e1 = erfinv_enclose(a_lo, a_hi)
    lo = 2*arb(SQPI)*arb(e0)*(arb(e0)**2).exp()
    hi = 2*arb(SQPI)*arb(e1)*(arb(e1)**2).exp()
    return float(lo.lower()), float(hi.upper())

def _core(a_lo, a_hi, b_lo, b_hi, which, deriv):
    q0,q1 = _Q(a_lo,a_hi); r0,r1 = _R(b_lo,b_hi)
    if not deriv:
        if which=='A': return q0*r0-1.0, q1*r1-1.0
        return q0*r0*(q0+r0)-6.0, q1*r1*(q1+r1)-6.0
    d0,d1 = _dQ(a_lo,a_hi)
    if which=='A': return d0*r0, d1*r1
    return d0*r0*(2*q0+r0), d1*r1*(2*q1+r1)

def _clamp(x): return min(max(x, EPS), 1.0-EPS)

def _seg(akind, blo, bhi, n, which, deriv, sq):
    """2*int of the core over b in (blo,bhi); sq=True uses b = bhi*t^2 with weight 2*bhi*t."""
    lo=hi=0.0
    for i in range(n):
        t0,t1 = i/n, (i+1)/n
        if sq: b0,b1 = bhi*t0*t0, bhi*t1*t1
        else:  b0,b1 = blo+(bhi-blo)*t0, blo+(bhi-blo)*t1
        b0,b1 = _clamp(b0), _clamp(b1)
        if b0>=b1: continue
        a0,a1 = akind(b0), akind(b1)
        if a0>a1: a0,a1 = a1,a0
        a0,a1 = _clamp(a0), _clamp(a1)
        f0,f1 = _core(a0,a1,b0,b1,which,deriv)
        if sq:
            w0,w1 = 2.0*bhi*t0, 2.0*bhi*t1
            c=[f0*w0,f0*w1,f1*w0,f1*w1]; dt=t1-t0
            lo += 2.0*dt*min(c); hi += 2.0*dt*max(c)
        else:
            db=b1-b0
            lo += 2.0*db*min(f0,f1); hi += 2.0*db*max(f0,f1)
    return lo,hi

def enclose_fn(r, n, which):
    m,M = 0.5*(1-r), 0.5*(1+r)
    lo=hi=0.0
    for akind, bl, bh, sq in ((lambda b: r-b,0.0,r,True),(lambda b: b-r,r,M,False),(lambda b: b+r,0.0,m,True)):
        a,b = _seg(akind,bl,bh,n,which,False,sq); lo+=a; hi+=b
    return lo,hi

def enclose_dfn(r, n, which):
    m,M = 0.5*(1-r), 0.5*(1+r)
    I1 = _seg(lambda b: r-b, 0.0, r, n, which, True, True)
    I2 = _seg(lambda b: b-r, r, M, n, which, True, False)
    I3 = _seg(lambda b: b+r, 0.0, m, n, which, True, True)
    Fmm = _core(_clamp(m),_clamp(m),_clamp(M),_clamp(M),which,False)   # F(m,M)
    FMM = _core(_clamp(M),_clamp(M),_clamp(m),_clamp(m),which,False)   # F(M,m)
    # alpha' = I1 + F(m,M) - I2 - F(M,m) + I3   (the _seg already carries the factor 2)
    lo = I1[0] + Fmm[0] - I2[1] - FMM[1] + I3[0]
    hi = I1[1] + Fmm[1] - I2[0] - FMM[0] + I3[1]
    return lo,hi

def enclose_W(r, n=2000):
    al = enclose_fn(r,n,'A');  be = enclose_fn(r,n,'B')
    da = enclose_dfn(r,n,'A'); db = enclose_dfn(r,n,'B')
    p1 = [x*y for x in al for y in db]
    p2 = [x*y for x in da for y in be]
    return min(p1)-max(p2), max(p1)-min(p2), al, be, da, db
