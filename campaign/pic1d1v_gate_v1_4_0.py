"""
pic1d1v_gate_v1_0_0.py — Thread-B P2 fidelity gate.

1D-1V periodic electrostatic PIC, immobile neutralizing ions, cold-lattice
linear stability: momentum-conserving (MCP) vs energy-conserving (ECP)
schemes with tent (b1) and quadratic (b2) spline kernels, tested against the
Finn-Evstatiev anchors (LANL_PIC_Finn.pdf, slides p27-p30). Gate targets
G1-G5 pre-registered in session_log.md, Session 17 / P2, 2026-08-03.

Theory arm: eigenvalues of the numerical Jacobian of the EXACT discrete
force map at the displaced lattice (deterministic; kernel/scheme-agnostic).
PIC arm: leapfrog time-domain with a deterministic multi-mode seed
perturbation; growth rate from a log-linear fit. NO RANDOM SEEDS CONSUMED.

Units: L = 1, eps0 = 1, m = 1, omega_p = 1 (q_e = -1/sqrt(Np)).

Author: James M. Hyman (Tulane) with Claude, Session 17, Thread B.
Date: 2026-08-05. Version 1.4.0. Change from v1_3_0: gate CSV filenames
carry the run mode (_FULL/_QUICK) so QUICK verification runs can never
overwrite FULL-mode data of record. Physics and gate logic unchanged. Changes per
slides p4/p13/p14/p16 rasters: (i) node-coincident lattice convention
(eps=0 puts particle alpha=0 on node i=0, matching p13); (ii) grid field
maps are SAMPLED CONTINUUM GREENS per p14/p16: E_i = Dg*sum_j G0(x_i-x_j)
rho_j with G0(x)=x-sgn(x)/2 (G0'=1-delta), and phi_i = Dg*sum_j L(x_i-x_j)
rho_j with L''=delta-1; in this code's charge convention ME=-Dg*G0mat,
Gphi=-Dg*Lmat (consistent: L'=-G0 => E=-phi'). Both arms use these maps. Change: analytic Jacobians (FD central
differences straddled kernel breakpoints at commensurate lattices, corrupting
G1 at eps=0 and faking ECP growth); midpoint convention at kernel jumps;
E-solve variants 'cent' (FD phi + centered E) and 'exact' (spectral Green). Change: neutralizing ions deposited from
the reference lattice (rho0=0, phi0=0, F(x_ref)=0 exactly), matching the
Finn-Evstatiev linearization; uniform-grid background caused a spurious
diagonal self-force term (v1_0_0 G1 lobe shape, false ECP growth).
"""
import sys, time
import numpy as np

FULL = "--full" in sys.argv          # QUICK smoke by default; --full = gate run
OUT = "/mnt/user-data/outputs"
EPS_FD = 1e-6                        # Jacobian central-difference step; positions O(1)

# ---------------------------------------------------------------- kernels ----
def w_tent(s):
    """b1 (tent) spline, W(s) = 1-|s| on |s|<1; sum_i W = 1 on any lattice."""
    a = np.abs(s); return np.where(a < 1.0, 1.0 - a, 0.0)

def dw_tent(s):
    """Derivative of tent with midpoint (distributional) values at jumps."""
    a = np.abs(s)
    out = np.where(a < 1.0, -np.sign(s), 0.0)
    out = np.where(np.isclose(a, 1.0), -0.5 * np.sign(s), out)
    return np.where(np.isclose(s, 0.0), 0.0, out)

def w_quad(s):
    """b2 (quadratic) spline: 3/4-s^2 (|s|<=1/2); (3/2-|s|)^2/2 (1/2<|s|<=3/2)."""
    a = np.abs(s)
    return np.where(a <= 0.5, 0.75 - s * s,
           np.where(a <= 1.5, 0.5 * (1.5 - a) ** 2, 0.0))

def dw_quad(s):
    a = np.abs(s)
    return np.where(a <= 0.5, -2.0 * s,
           np.where(a <= 1.5, -np.sign(s) * (1.5 - a), 0.0))

KER = {"tent": (w_tent, dw_tent), "quad": (w_quad, dw_quad)}

# ------------------------------------------------------------- force maps ----
class Pic1d:
    """Exact discrete force map for the periodic 1D ES system.

    Parameters: Ng cells (nodes x_i = i*Dg), Np particles, kernel name,
    scheme "mcp" (deposit-W, spectral FD-Laplacian solve, centered E,
    gather-W) or "ecp" (same solve; force = -grad of U = (1/2) rho.phi Dg,
    i.e. gather with W'). eps0=1, m=1; q_e = -1/sqrt(Np) so omega_p = 1.
    """
    def __init__(self, Ng, Np, kernel="tent", scheme="mcp", x_ref=None):
        self.Ng, self.Np, self.L = Ng, Np, 1.0
        self.Dg = self.L / Ng
        self.qe = -1.0 / np.sqrt(Np)          # omega_p^2 = n0 qe^2/(eps0 m) = 1
        self.rho_bg = -self.qe * Np / self.L  # uniform fallback (legacy)
        self.rho_bg_grid = None               # lattice-deposited ions (preferred)
        self.W, self.dW = KER[kernel]
        self.scheme = scheme
        self.xi = np.arange(Ng) * self.Dg
        self.ME, self.Gphi = grid_maps(Ng, self.Dg, "samp")

    def set_ref(self, x_ref):
        """Freeze neutralizing ions as the mirror deposit of x_ref (rho0 = 0)."""
        s = self._s(np.asarray(x_ref))
        self.rho_bg_grid = -(self.qe / self.Dg) * self.W(s).sum(axis=0)

    def _s(self, x):
        d = x[:, None] - self.xi[None, :]
        d -= self.L * np.round(d / self.L)    # periodic minimum image
        return d / self.Dg                    # (Np, Ng) normalized separations

    def fields(self, x):
        s = self._s(x)
        Wm = self.W(s)
        bg = self.rho_bg_grid if self.rho_bg_grid is not None else self.rho_bg
        rho = (self.qe / self.Dg) * Wm.sum(axis=0) + bg
        phi = self.Gphi @ rho
        return s, Wm, rho, phi

    def force(self, x):
        """Force per unit mass on each particle (m = 1)."""
        s, Wm, rho, phi = self.fields(x)
        if self.scheme == "mcp":
            E = self.ME @ rho
            return self.qe * (Wm * E[None, :]).sum(axis=1)
        # ECP: F_p = -dU/dx_p, U = (1/2) sum_i rho_i phi_i Dg  (exact gradient)
        return -(self.qe / self.Dg) * (self.dW(s) * phi[None, :]).sum(axis=1)

    def lattice(self, eps_frac):
        """Uniform lattice shifted by eps = eps_frac * Dp, Dp = L/Np."""
        Dp = self.L / self.Np      # node-coincident at eps=0 (slide p13)
        return (np.arange(self.Np)) * Dp + eps_frac * Dp

# ------------------------------------------------------------- theory arm ----
def grid_maps(Ng, Dg, esolve):
    """Return (ME, Gphi): linear maps rho -> E and rho -> phi on the grid.

    esolve='cent': FD-Laplacian phi-solve + centered-difference E (BL grid).
    esolve='exact': continuum Green sampled spectrally (meshfree heritage):
    phi_k = rho_k/(2 pi m)^2, E_k = -i k phi_k.
    """
    if esolve == "samp":
        d = (np.arange(Ng)[:, None] - np.arange(Ng)[None, :]) * Dg
        d -= np.round(d)                      # minimum image on L = 1
        G0 = np.where(np.isclose(d, 0.0), 0.0, d - 0.5 * np.sign(d))
        Lm = 0.5 * np.abs(d) - 0.5 * d * d - 1.0 / 12.0
        return -Dg * G0, -Dg * Lm             # ME, Gphi (this convention)
    m = np.fft.fftfreq(Ng, d=1.0 / Ng)
    if esolve == "cent":
        k2 = (2.0 * np.sin(np.pi * m / Ng) / Dg) ** 2
    else:
        k2 = (2.0 * np.pi * m) ** 2
    inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
    F = np.fft.fft(np.eye(Ng), axis=0)
    Fi = np.fft.ifft(np.eye(Ng), axis=0)
    Gphi = np.real(Fi @ (inv[:, None] * F))
    if esolve == "cent":
        D0 = (np.roll(np.eye(Ng), -1, 0) - np.roll(np.eye(Ng), 1, 0)) / (2 * Dg)
        ME = -D0 @ Gphi
    else:
        ik = 1j * 2.0 * np.pi * m
        ME = np.real(Fi @ ((-ik * inv)[:, None] * F))
    return ME, Gphi


def theory_omega(Ng, Nppc, kernel, scheme, eps_frac, esolve="samp"):
    """Complex eigenfrequencies from the ANALYTIC Jacobian at the exact
    equilibrium (rho0 = 0, E0 = 0):
      MCP: J = (qe^2/Dg^2) W  ME  W'^T   (asymmetric -> can be unstable)
      ECP: J = -(qe^2/Dg^2) W' Gphi W'^T (symmetric NSD -> stable)
    Returns (gamma, omega), gamma = max Im omega (amplitude convention).
    """
    p = Pic1d(Ng, Ng * Nppc, kernel, scheme)
    x0 = p.lattice(eps_frac)
    sm = p._s(x0)
    Wm, Wpm = p.W(sm), p.dW(sm)
    ME, Gphi = grid_maps(Ng, p.Dg, esolve)
    c = p.qe ** 2 / p.Dg ** 2
    if scheme == "mcp":
        J = c * (Wm @ ME @ Wpm.T)
    else:
        J = -c * (Wpm @ Gphi @ Wpm.T)
    lam = np.linalg.eigvals(-J)
    om = np.sqrt(lam.astype(complex))
    return float(np.max(np.abs(om.imag))), om

# ----------------------------------------------------------------- PIC arm ----
def pic_growth(Ng, Nppc, kernel, scheme, eps_frac, gamma_hint,
               dt=0.05, amp0=1e-7):
    """Leapfrog growth-rate measurement with a deterministic seed.

    Perturbs the lattice with fixed-phase sinusoids on all modes, evolves
    until RMS displacement grows ~e^6 (capped), fits ln RMS(t) on the clean
    exponential window [10*amp0, 1e-3]. Returns fitted gamma (amplitude).
    """
    p = Pic1d(Ng, Ng * Nppc, kernel, scheme)
    x0 = p.lattice(eps_frac)
    p.set_ref(x0)                             # ions frozen at the lattice
    modes = np.arange(1, Ng // 2 + 1)
    pert = sum(np.sin(2 * np.pi * m * x0 / p.L + 0.7 * m) for m in modes)
    x = x0 + amp0 * pert / np.max(np.abs(pert))
    v = np.zeros_like(x)
    T = min(6.0 / max(gamma_hint, 1e-3), 4000.0)
    nsteps = int(T / dt)
    v += 0.5 * dt * p.force(x)                # leapfrog half-kick
    ts, amps = [], []
    for k in range(nsteps):
        x = (x + dt * v) % p.L
        v += dt * p.force(x)
        if k % 5 == 0:
            d = x - x0; d -= p.L * np.round(d / p.L)
            ts.append((k + 1) * dt); amps.append(float(np.sqrt(np.mean(d * d))))
    ts, amps = np.array(ts), np.array(amps)
    lo, hi = 10 * amp0, 1e-3
    msk = (amps > lo) & (amps < hi)
    if msk.sum() < 8:                         # convergence guard
        return float("nan")
    return float(np.polyfit(ts[msk], np.log(amps[msk]), 1)[0])

# ---------------------------------------------------------- verification -----
def verify():
    print("=" * 66); print("VERIFICATION SUITE"); print("=" * 66)
    rng_s = np.linspace(-2.1, 2.1, 7)[None, :] * 0 + \
        (np.linspace(0.0, 0.99, 5)[:, None] + np.arange(-3, 4)[None, :])
    for name, (W, _) in KER.items():          # V1: partition of unity
        tot = W(rng_s).sum(axis=1)
        assert np.allclose(tot, 1.0, atol=1e-12), f"V1 FAIL {name}"
    print("V1 PASS  sum_i W(x - x_i) = 1 for tent and quad (5 offsets)")
    p = Pic1d(16, 48, "tent", "mcp")          # V2: charge neutrality
    p.set_ref(p.lattice(0.31))
    _, _, rho, _ = p.fields(p.lattice(0.31))
    assert np.max(np.abs(rho)) < 1e-12, "V2b FAIL rho0 not identically 0"
    assert abs(rho.mean()) < 1e-12, "V2 FAIL neutrality"
    print("V2 PASS  rho0 = 0 identically at the reference lattice (exact equilibrium)")
    F = p.force(p.lattice(0.31) + 1e-4 * np.sin(
        2 * np.pi * 3 * p.lattice(0.31)))     # V3: MCP momentum conservation
    assert abs(F.sum()) < 1e-10, "V3 FAIL momentum"
    print("V3 PASS  MCP: sum_p F_p = 0 (momentum conserving)")
    pe = Pic1d(8, 24, "quad", "ecp")          # V4: ECP force = -grad U exactly
    pe.set_ref(pe.lattice(0.2))
    x = pe.lattice(0.2) + 1e-3 * np.cos(2 * np.pi * 2 * pe.lattice(0.2))
    def U(y):
        _, _, r, ph = pe.fields(y); return 0.5 * float((r * ph).sum()) * pe.Dg
    g = np.array([(U(x + h * e) - U(x - h * e)) / (2 * h)
                  for h, e in ((1e-6, np.eye(24)[j]) for j in range(24))])
    assert np.allclose(pe.force(x), -g, atol=1e-6), "V4 FAIL ECP gradient"
    print("V4 PASS  ECP force equals -grad U (exact energy gradient)")
    print("All 4 verification tests PASSED."); print("=" * 66)

# ------------------------------------------------------------- gate suite ----
def run_gate():
    t0 = time.time(); res = {}
    MODE = "FULL" if FULL else "QUICK"
    nE = 21 if FULL else 11
    eps_grid = np.linspace(0.0, 1.0, nE)

    # G1/G2/G4: eps-scans at Ng=8, Nppc=3
    for tag, kern, sch in (("G1", "tent", "mcp"), ("G2", "quad", "mcp"),
                           ("G4", "tent", "ecp")):
        gam = np.array([theory_omega(8, 3, kern, sch, e)[0] for e in eps_grid])
        np.savetxt(f"{OUT}/gate_{tag}_epsscan_{kern}_{sch}_{MODE}.csv",
                   np.column_stack([eps_grid, gam]), delimiter=",",
                   header="eps_over_Dp,gamma", comments="")
        res[tag] = gam
    # Commensuration limit: exact eps=0 is a removable point (W'_tent evaluated
    # at its jump with the midpoint convention); the anchor quantity is the
    # eps -> 0 limit, evaluated here at eps = 1e-3. gamma(exact 0) reported too.
    res["G1_lim0"] = theory_omega(8, 3, "tent", "mcp", 1e-3)[0]
    res["G1_at0"] = res["G1"][0]
    _, om0 = theory_omega(8, 3, "tent", "mcp", 0.02)
    rebands = np.unique(np.round(np.abs(om0.real), 2))
    res["G1_re"] = rebands

    # G3: Nppc slopes at Ng=16
    nppc = np.array([1, 2, 3, 4, 6, 8] + ([12, 16] if FULL else []))
    slopes = {}
    for kern in ("tent", "quad"):
        for ef in (0.15, 0.25):
            g = np.array([theory_omega(16, n, kern, "mcp", ef)[0] for n in nppc])
            np.savetxt(f"{OUT}/gate_G3_nppc_{kern}_eps{ef}_{MODE}.csv",
                       np.column_stack([nppc, g]), delimiter=",",
                       header="Nppc,gamma", comments="")
            m = g > 1e-12
            slopes[(kern, ef)] = float(np.polyfit(
                np.log(nppc[m][1:]), np.log(g[m][1:]), 1)[0])
    res["G3"] = slopes

    # G5: Ng plateaus, tent, eps=0.25
    ngs = [16, 64, 256] if FULL else [16, 64]
    g5 = {n: [theory_omega(ng, n, "tent", "mcp", 0.25)[0] for ng in ngs]
          for n in (1, 2, 3)}
    res["G5"] = (ngs, g5)

    # PIC spot checks (deterministic)
    spots = [("tent", "mcp", 0.001), ("tent", "mcp", 0.25), ("quad", "mcp", 0.25)]
    pic = {}
    for kern, sch, ef in spots:
        gth = theory_omega(8, 3, kern, sch, ef)[0]
        pic[(kern, ef)] = (gth, pic_growth(8, 3, kern, sch, ef, gth))
    res["PIC"] = pic

    # ------------------------------------------------------------ verdicts ---
    print(f"\nGATE RESULTS  ({'FULL' if FULL else 'QUICK'} mode, "
          f"{time.time()-t0:.1f}s)")
    g1 = res["G1"]; i0 = np.argmin(np.abs(eps_grid - 0.5))
    g1max = max(g1.max(), res["G1_lim0"])
    ok1 = (abs(g1max - 0.15) / 0.15 < 0.15) and (g1[i0] < 0.10 * g1max) \
        and (res["G1_lim0"] / g1max > 0.95)
    print(f"G1 tent eps-scan: gamma(eps->0)={res['G1_lim0']:.4f} "
          f"(target 0.15+/-15%; gamma(exact 0)={res['G1_at0']:.1e}, removable), "
          f"gamma(Dp/2)={g1[i0]:.2e}, Re bands={res['G1_re']}"
          f"  -> {'PASS' if ok1 else 'FAIL'}")
    g2 = res["G2"]; g2max = g2.max()
    ok2 = (abs(g2max - 0.003) / 0.003 < 0.15) and (g2[0] < 0.15 * g2max) \
        and (g2[i0] < 0.15 * g2max)
    print(f"G2 quad eps-scan: gamma_max={g2max:.5f} (target 0.003+/-15%), "
          f"gamma(0)={g2[0]:.1e}, gamma(Dp/2)={g2[i0]:.1e}"
          f"  -> {'PASS' if ok2 else 'FAIL'}")
    s = res["G3"]; okt = all(abs(s[('tent', e)] + 1) < 0.15 for e in (0.15, 0.25))
    okq = all(abs(s[('quad', e)] + 3) < 0.30 for e in (0.15, 0.25))
    print(f"G3 slopes: tent {s[('tent',0.15)]:.2f}/{s[('tent',0.25)]:.2f} "
          f"(target -1), quad {s[('quad',0.15)]:.2f}/{s[('quad',0.25)]:.2f} "
          f"(target -3)  -> {'PASS' if okt and okq else 'FAIL'}")
    g4 = res["G4"]; ok4 = g4.max() < 1e-6
    print(f"G4 ECP: max gamma = {g4.max():.2e} (target < 1e-6)"
          f"  -> {'PASS' if ok4 else 'FAIL'}")
    ngs, g5 = res["G5"]; tgt5 = {1: 0.20, 2: 0.11, 3: 0.077}
    ok5 = all(abs(g5[n][-1] - tgt5[n]) / tgt5[n] < 0.10 for n in (1, 2, 3))
    print(f"G5 plateaus at Ng={ngs[-1]}: " + ", ".join(
        f"Nppc={n}: {g5[n][-1]:.3f} (tgt {tgt5[n]})" for n in (1, 2, 3))
        + f"  -> {'PASS' if ok5 else 'FAIL (secondary)'}")
    for (kern, ef), (gth, gp) in res["PIC"].items():
        rel = abs(gp - gth) / gth if gth > 0 else float("nan")
        print(f"PIC spot {kern} eps={ef}: theory {gth:.4f}, "
              f"PIC {gp:.4f}, rel {rel:.1%}"
              f"  -> {'PASS' if rel < 0.20 else 'CHECK'}")
    core = ok1 and ok2 and okt and okq and ok4
    print(f"\nCORE GATE (G1-G4): {'PASS' if core else 'FAIL'}")
    return core

if __name__ == "__main__":
    verify()
    run_gate()
