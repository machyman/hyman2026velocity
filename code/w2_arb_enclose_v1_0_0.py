"""Rigorous enclosure of alpha(r) and beta(r) by monotone-corner subdivision.

No derivatives are used, so the singular dQ ~ exp(erfinv^2) never appears.
Q(p) = 2 erfinv(p)^2 is increasing on (0,1); R(b) = 2 erfcinv(b)^2 is
decreasing.  Both are positive.  Over a box in (a,b) the product QR is
therefore bracketed by its corner values, which is what makes the enclosure
cheap and sound.

Endpoints: the substitution b = hi*t^2 regularises the logarithmic endpoint at
b -> 0, exactly as the float code does, so the substituted integrand vanishes
there and the first subinterval carries a finite enclosure.
"""
from w2_arb_erfinv_v1_0_0 import Q_enclose, R_enclose

def _F_box(a_lo, a_hi, b_lo, b_hi, const):
    """Enclose F = Q(a)R(b) - const over the box."""
    qlo, qhi = Q_enclose(a_lo, a_hi)
    rlo, rhi = R_enclose(b_lo, b_hi)
    return qlo*rlo - const, qhi*rhi - const

def _FB_box(a_lo, a_hi, b_lo, b_hi):
    """Enclose FB = QR(Q+R) - 6 over the box (all factors positive, increasing in Q and R)."""
    qlo, qhi = Q_enclose(a_lo, a_hi)
    rlo, rhi = R_enclose(b_lo, b_hi)
    return qlo*rlo*(qlo+rlo) - 6.0, qhi*rhi*(qhi+rhi) - 6.0

def _seg_sq(kind, r, hi, n, const, EPS):
    """2 * int_0^hi F(a(b), b) db with b = hi t^2, weight 2 hi t.  kind gives a(b)."""
    lo_s = hi_s = 0.0
    for i in range(n):
        t0, t1 = i/n, (i+1)/n
        b0, b1 = hi*t0*t0, hi*t1*t1
        b0 = max(b0, EPS); b1 = max(b1, b0+EPS)
        if b1 >= 1.0: b1 = 1.0-EPS
        if b0 >= b1: continue
        a0, a1 = kind(b1), kind(b0)          # a decreasing in b for 'r-b'
        if a0 > a1: a0, a1 = a1, a0
        a0 = min(max(a0, EPS), 1.0-EPS); a1 = min(max(a1, EPS), 1.0-EPS)
        if a0 > a1: continue
        f0, f1 = (_F_box(a0,a1,b0,b1,const) if const==1.0 else _FB_box(a0,a1,b0,b1))
        w0, w1 = 2.0*hi*t0, 2.0*hi*t1        # weight 2*hi*t, increasing
        dt = t1-t0
        cands = [f0*w0, f0*w1, f1*w0, f1*w1]
        lo_s += 2.0*dt*min(cands); hi_s += 2.0*dt*max(cands)
    return lo_s, hi_s

def _seg_lin(kind, lo, hi, n, const, EPS):
    """2 * int_lo^hi F(a(b), b) db, no singular endpoint."""
    lo_s = hi_s = 0.0
    for i in range(n):
        b0 = lo + (hi-lo)*i/n; b1 = lo + (hi-lo)*(i+1)/n
        b0 = max(b0, EPS); b1 = min(b1, 1.0-EPS)
        if b0 >= b1: continue
        a0, a1 = kind(b0), kind(b1)
        if a0 > a1: a0, a1 = a1, a0
        a0 = min(max(a0, EPS), 1.0-EPS); a1 = min(max(a1, EPS), 1.0-EPS)
        f0, f1 = (_F_box(a0,a1,b0,b1,const) if const==1.0 else _FB_box(a0,a1,b0,b1))
        lo_s += 2.0*(b1-b0)*f0; hi_s += 2.0*(b1-b0)*f1
    return lo_s, hi_s

def enclose(r, n=400, which='A', EPS=1e-14):
    const = 1.0 if which=='A' else 6.0
    m, M = 0.5*(1.0-r), 0.5*(1.0+r)
    lo = hi = 0.0
    for seg in (
        ('sq', (lambda b: r-b), r, None),
        ('lin',(lambda b: b-r), (r, M), None),
        ('sq', (lambda b: b+r), m, None),
    ):
        if seg[0]=='sq':
            a,b = _seg_sq(seg[1], r, seg[2], n, const, EPS)
        else:
            a,b = _seg_lin(seg[1], seg[2][0], seg[2][1], n, const, EPS)
        lo += a; hi += b
    return lo, hi
