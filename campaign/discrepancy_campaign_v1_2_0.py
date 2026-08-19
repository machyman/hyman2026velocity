"""Discrepancy campaign harness — Session 18 / P2, Thread B (Array-RQMC x PIC).

Purpose: measure the joint (x,v) kernel discrepancy D_J(t) of the particle
ensemble vs f_eq = U(0,1) x N(0, v_th^2) for the r3-registered arms in the
collision-dominated regime (nu*dt = 1), with the x-only D_K2 recorded as a
secondary diagnostic. MCP cold companions retain the x-metric and G5 gamma
anchors.

Key claims tested (registered 2026-08-05 + r1/r2 + r3 amendment,
session_log.md): C1 window-mean within +/-15% of D_MC_J; C2 (no collisions,
kinematic characterization) in [0.7,1.1]*D_MC_J; C3 at plateau, +/-25%;
C4a/C4b window-mean <= 0.5*D_MC_J with ladder slope in [-1.05,-0.65].

Baselines of record: A1-J analytic iid plateau D_MC_J = 6.5064/sqrt(Np)
(Ng=64, h_v=v_th/2); free-streaming envelope (deterministic): bit-reversal
Np=1024 D_J/D_MC_J = 0.167 (t=0), 0.855 (t=5), 0.929 (t=80); A3 gate G5
gammas .204/.112/.077.

Physics is IMPORTED from the hash-locked fidelity gate:
  pic1d1v_gate_v1_4_0.py, sha256-16 dfa9da580ba210ca (FULL GATE PASS).

Seeds: BASE 20260803; stream = default_rng(SeedSequence(BASE, spawn_key=
(offset,))). Registered map enforced (FULL): C1 22-26, C2 27-31, C3 32-37,
C4a 38-43, C4b 44-49, collisional-MCP 50-56, CTRL 57-59, spares 60-61;
smoke (QUICK) offsets 0-1 only. C2-MCP deterministic, consumes no offsets.

Author:  James M. Hyman (Tulane) with Claude — Session 18, Thread B.
Date:    2026-08-07.  Version 1.2.0 (implements the r4 registration of
2026-08-07; session_log entry of record).
r4 additions: kac_rotate_monotone for M4a/M4b (R1: iid Rademacher partner
sign from the offset stream; R2: particle a = sort-matched pair member;
exact joint law of the full-circle rotation, v_b' = sg*R*sin(pi*u));
snapshot arms S-C4b (rotation, unchanged physics) and S-C3 (iid control),
Np in {2048, 8192}, npz at t in {0, 10, 20, 40}, crash-safe; offset block
62-81 (M4a 62-67 | M4b 68-73 | S-C4b 74-76 | S-C3 77-79 | spares 80-81);
r3 arms locked out of campaign mode (offsets consumed, results of
record); V9 distributional-equivalence test (tolerances fixed in code
and printed); verification suite = 10 tests.
Patch 1.1.1: resume check requires the r3 CSV header (t,DJ,Dx) so files
with a stale superseded schema can never resume-skip a registered run
(finding S18-F5).
Changes from v1.0.1: joint metric D_J primary (real-space K2, Gaussian
v-kernel h_v = v_th/2, exact pairwise, row-blocked); Kac pair rotation
replaces swap, angle th = 2*pi*u (C3: stream iid; C4: matched Sobol' col 3;
CTRL: shuffled row assignment), followed by exact global P-projection and
KE restoration; nu = 20 (nu*dt = 1), T_ECP = 40; CSVs carry t,DJ,Dx;
V-suite extended to 9 tests (8 registered r3 + carried Hilbert exactness).
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pic1d1v_gate_v1_4_0 as gate  # sha256-16 dfa9da580ba210ca

# ── Frozen configuration (r1/r2 carried; r3 changes marked) ──────────────────
BASE_SEED = 20260803
NG   = 64
DG   = 1.0 / NG
VTH  = 0.05                 # lambda_D = 3.2*Dg
HV   = VTH / 2.0            # r3: joint-metric v bandwidth [DEFAULT]
NU   = 20.0                 # r3: nu*dt = 1, collision-dominated [DEFAULT]
DT   = 0.05
A0   = 1e-6
EPS_MCP = 0.25
MMAX = 1024
T_ECP, T_MCP = 40.0, 160.0  # r3: ECP T=40, window [20,40]
SAMPLE_EVERY = int(round(1.0 / DT))
KTIL2_0 = 2.0 / (3.0 * DG) - 1.0
K2_ZERO = 2.0 / (3.0 * DG)

LADDER_ECP = (2, 4, 8, 16, 32, 64, 128)
LADDER_MCP = (1, 2, 3)
OFFSETS = {("C1", "ecp"): range(22, 27), ("C2", "ecp"): range(27, 32),
           ("C3", "ecp"): range(32, 38), ("C4a", "ecp"): range(38, 44),
           ("C4b", "ecp"): range(44, 50), ("CTRL", "ecp"): range(57, 60),
           # r4 (registered 2026-08-07): block 62-81
           ("M4a", "ecp"): range(62, 68), ("M4b", "ecp"): range(68, 74),
           ("S-C4b", "ecp"): range(74, 77), ("S-C3", "ecp"): range(77, 80)}
R4_ARMS = ("M4a", "M4b", "S-C4b", "S-C3")
LADDER_S = (32, 128)                    # r4 S-arms: Np in {2048, 8192}
SNAP_TIMES = (0.0, 10.0, 20.0, 40.0)    # r4 snapshot grid
MCP_COLL_OFFSETS = range(50, 57)
SMOKE_OFFSETS = (0, 1)

# Free-streaming envelope of record (r3 scratch; deterministic, 3 dp):
FS_ENVELOPE = {0.0: 0.167, 5.0: 0.855, 80.0: 0.929}   # Np=1024, bit-reversal

_M = np.arange(1, MMAX + 1)
_W4 = np.sinc(_M * DG) ** 4
A1_TRUNC = 2.0 * float(_W4.sum())


def rng_for(offset):
    """Frozen stream (r2): default_rng(SeedSequence(BASE, spawn_key=(offset,)))."""
    return np.random.default_rng(np.random.SeedSequence(BASE_SEED,
                                                        spawn_key=(offset,)))


# ── Metrics ──────────────────────────────────────────────────────────────────
def d_k2(x):
    """Secondary diagnostic: x-only K2 discrepancy (r2 Fourier form, M=1024)."""
    npart = x.size
    ph = np.exp(2j * np.pi * np.outer(_M, x)).sum(axis=1)
    s = (ph.real ** 2 + ph.imag ** 2) / npart ** 2
    return float(np.sqrt(2.0 * np.dot(s, _W4)))


def d_mc(npart):
    """A1 analytic x-only iid plateau."""
    return float(np.sqrt(KTIL2_0 / npart))


def K2_real(r):
    """Real-space K2 = W*W for the unit-mass tent, torus min-image.

    K2(r) = (1/Dg)*k(|r|/Dg): k(s) = s^3/2 - s^2 + 2/3 on [0,1],
    (2-s)^3/6 on [1,2], 0 beyond; K2(0) = 2/(3*Dg).
    """
    s = np.minimum(np.abs(r), 1.0 - np.abs(r)) / DG
    out = np.zeros_like(s)
    m1 = s <= 1.0
    m2 = (s > 1.0) & (s < 2.0)
    out[m1] = s[m1] ** 3 / 2 - s[m1] ** 2 + 2.0 / 3.0
    out[m2] = (2.0 - s[m2]) ** 3 / 6.0
    return out / DG


def d_joint(x, v, h=HV, block=512):
    """r3 primary metric: exact pairwise MMD vs f_eq, row-blocked for memory.

    D_J^2 = (1/Np^2) sum_ab K2(dx) exp(-dv^2/2h^2) - (2/Np) sum_a g(v_a)
            + h/sqrt(h^2 + 2 s^2),  s = VTH,
    g(v) = (h/hypot(h,s)) exp(-v^2/(2(h^2+s^2))).
    """
    n = x.size
    term1 = 0.0
    for i0 in range(0, n, block):
        dx = x[i0:i0 + block, None] - x[None, :]
        dv = v[i0:i0 + block, None] - v[None, :]
        term1 += float((K2_real(dx) * np.exp(-dv * dv / (2 * h * h))).sum())
    term1 /= n * n
    g = (h / np.hypot(h, VTH)) * np.exp(-v * v / (2 * (h * h + VTH * VTH)))
    term3 = h / np.sqrt(h * h + 2 * VTH * VTH)
    return float(np.sqrt(max(term1 - 2.0 * float(g.mean()) + term3, 0.0)))


def d_mc_joint(npart, h=HV):
    """A1-J analytic joint iid plateau: sqrt((K2(0) - h/sqrt(h^2+2s^2))/Np)."""
    return float(np.sqrt((K2_ZERO - h / np.sqrt(h * h + 2 * VTH * VTH)) / npart))


# ── Loads (frozen) ───────────────────────────────────────────────────────────
def bit_reverse_perm(n):
    b = n.bit_length() - 1
    j = np.arange(n)
    r = np.zeros(n, dtype=np.int64)
    for k in range(b):
        r |= ((j >> k) & 1) << (b - 1 - k)
    return r


def load_iid(npart, rng):
    return rng.random(npart), VTH * rng.standard_normal(npart)


def load_quiet(npart):
    j = np.arange(npart)
    x = (j + 0.5) / npart
    v = VTH * ndtri((bit_reverse_perm(npart) + 0.5) / npart)
    return x, v


def load_mcp_cold(p, npart):
    """Registered MCP companion load (gate-mirrored, deterministic)."""
    x0 = p.lattice(EPS_MCP)
    p.set_ref(x0)
    modes = np.arange(1, NG // 2 + 1)
    pert = sum(np.sin(2 * np.pi * m * x0 / p.L + 0.7 * m) for m in modes)
    x = x0 + A0 * pert / np.max(np.abs(pert))
    return x % p.L, np.zeros(npart)


# ── Collision machinery (r3: Kac rotations + invariant projections) ──────────
def kac_rotate(v, pairs, u):
    """Kac pair rotation, th = 2*pi*u per fired pair; conserves pair KE."""
    if len(pairs):
        th = 2.0 * np.pi * u
        c, s = np.cos(th), np.sin(th)
        a, b = pairs[:, 0], pairs[:, 1]
        va, vb = v[a].copy(), v[b].copy()
        v[a] = va * c + vb * s
        v[b] = -va * s + vb * c


def kac_rotate_monotone(v, pairs, u, sg):
    """r4 M-arms (registered 2026-08-07): exact-law monotone coupling.

    Sort-matched member a (pairs[:, 0], mirroring the r3 row assignment;
    registration R2) gets v_a' = -R*cos(pi*u), strictly increasing in u
    with the exact arcsine marginal of the full-circle rotation; partner
    gets v_b' = sg*R*sin(pi*u), sg an iid Rademacher sign from the offset
    stream (registration R1). Joint law equals the full-circle rotation;
    pair KE conserved to rounding (cancellation-free form).
    """
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        rad = np.hypot(v[a], v[b])
        arg = np.pi * u
        v[a] = -rad * np.cos(arg)
        v[b] = sg * rad * np.sin(arg)


def project_invariants(v, ke0):
    """r3 frozen: exact global P removal, then exact KE restoration."""
    v -= v.mean()
    ke = 0.5 * float(np.dot(v, v))
    if ke > 0.0:
        v *= np.sqrt(ke0 / ke)


def n_events(acc):
    nev = int(acc)
    return nev, acc - nev


def _adjacent_pairs(order):
    m = (len(order) // 2) * 2
    return order[:m].reshape(-1, 2)


def _rqmc_select(pairs, nev, rng, shuffle=False):
    """r3 firing+angle rule: fresh scrambled Sobol' d=3; col 1 sorted <->
    pair rank; pair r fires iff u_r2 < nev/N_pairs; col 3 supplies the Kac
    angle variate u. shuffle=True permutes row assignment (CTRL)."""
    npairs = len(pairs)
    if npairs == 0 or nev == 0:
        return pairs[:0], np.empty(0)
    eng = qmc.Sobol(d=3, scramble=True, seed=rng)
    pts = eng.random_base2(int(np.ceil(np.log2(max(npairs, 2)))))[:npairs]
    pts = pts[np.argsort(pts[:, 0])]
    if shuffle:
        rng.shuffle(pts, axis=0)
    fire = pts[:, 1] < nev / npairs
    return pairs[fire], pts[fire, 2]


def pairs_iid(npart, nev, rng):
    """C3: random disjoint pairs + iid angle variates from the stream."""
    pr = rng.permutation(npart)[:2 * min(nev, npart // 2)].reshape(-1, 2)
    return pr, rng.random(len(pr))


def pairs_vsort_cell(x, v, nev, rng):
    """C4a: per-cell v-sort adjacent-rank pairs (cross-cell dropped)."""
    cell = np.minimum((x * NG).astype(np.int64), NG - 1)
    order = np.lexsort((v, cell))
    p = _adjacent_pairs(order)
    return _rqmc_select(p[cell[p[:, 0]] == cell[p[:, 1]]], nev, rng)


def hilbert_xy2d(px, py, order=8):
    """Vectorized Hilbert index, (x,y) orientation; bit-exact vs reference."""
    x = px.astype(np.int64).copy()
    y = py.astype(np.int64).copy()
    d = np.zeros_like(x)
    s = 1 << (order - 1)
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        m = ry == 0
        xm, ym, rxm = x[m], y[m], rx[m]
        x[m] = np.where(rxm == 1, s - 1 - ym, ym)
        y[m] = np.where(rxm == 1, s - 1 - xm, xm)
        s >>= 1
    return d


def _hilbert_key(x, v):
    side = (1 << 8) - 1
    u = np.clip((np.stack([x % 1.0, ndtr(v / VTH)], axis=1) * side)
                .astype(np.int64), 0, side)
    return hilbert_xy2d(u[:, 0], u[:, 1])


def pairs_hilbert(x, v, nev, rng, shuffle=False):
    """C4b/CTRL: global (x,v) Hilbert-key adjacent-rank pairs."""
    order = np.argsort(_hilbert_key(x, v), kind="stable")
    return _rqmc_select(_adjacent_pairs(order), nev, rng, shuffle=shuffle)


# ── Offset discipline ────────────────────────────────────────────────────────
def _check_offset(arm, scheme, offset, mode, collide):
    if mode == "QUICK":
        assert offset in SMOKE_OFFSETS, \
            f"QUICK runs use smoke offsets {SMOKE_OFFSETS} only (got {offset})"
        return
    if scheme == "ecp":
        ok = OFFSETS.get((arm, scheme))
        assert ok is not None and offset in ok, \
            f"offset {offset} not registered for ({arm},{scheme})"
    else:
        if collide:
            assert offset in MCP_COLL_OFFSETS, \
                f"collisional MCP offsets are {MCP_COLL_OFFSETS} (got {offset})"
        else:
            assert arm == "C2", "non-collisional MCP companion is C2 only"


# ── Dynamics ─────────────────────────────────────────────────────────────────
def _csv_path(outdir, arm, scheme, npart, offset, mode):
    return os.path.join(outdir,
                        f"discrepancy_{arm}_{scheme}_Np{npart}_o{offset}_{mode}.csv")


def _complete(path, t_end):
    if not os.path.isfile(path):
        return False
    try:
        lines = open(path).read().rstrip().splitlines()
        if not lines or lines[0].strip() != "t,DJ,Dx":
            return False
        return float(lines[-1].split(",")[0]) >= t_end - 1e-9
    except (ValueError, IndexError):
        return False


def _write_snapshot(outdir, arm, npart, offset, mode, tval, x, v):
    """r4 S-arms: crash-safe npz snapshot (leapfrog-staggered v, matching
    the metric convention)."""
    tag = f"snapshot_{arm}_Np{npart}_o{offset}_t{int(round(tval))}"
    if mode == "QUICK":
        tag += "_QUICK"
    np.savez(os.path.join(outdir, tag + ".npz"), x=x, v=v)


def evolve(arm, scheme, npart, offset, t_end, mode, outdir=None):
    """Leapfrog + gate field maps; r3 Kac collisions; metrics every 1/omega_p.

    Returns (t, DJ, Dx) arrays, or (None, None, None) on resume-skip.
    """
    collide = arm in ("C3", "C4a", "C4b", "CTRL",
                      "M4a", "M4b", "S-C4b", "S-C3")
    _check_offset(arm, scheme, offset, mode, collide)
    fn = _csv_path(outdir, arm, scheme, npart, offset, mode) if outdir else None
    if fn and _complete(fn, t_end):
        print(f"  resume-skip {os.path.basename(fn)}")
        return None, None, None
    rng = rng_for(offset)
    p = gate.Pic1d(NG, npart, "tent", scheme)
    if scheme == "mcp":
        x, v = load_mcp_cold(p, npart)
    else:
        p.set_ref(p.lattice(0.0))
        x, v = load_iid(npart, rng) if arm == "C1" else load_quiet(npart)
    nsteps = int(round(t_end / DT))
    acc = 0.0
    v = v + 0.5 * DT * p.force(x)
    writer = open(fn, "w", buffering=1) if fn else None
    if writer:
        writer.write("t,DJ,Dx\n")
    ts, djs, dxs = [0.0], [d_joint(x, v)], [d_k2(x)]
    if writer:
        writer.write(f"0.0,{djs[0]:.10e},{dxs[0]:.10e}\n")
    if arm.startswith("S-") and outdir:
        _write_snapshot(outdir, arm, npart, offset, mode, 0.0, x, v)
    for k in range(1, nsteps + 1):
        x = (x + DT * v) % p.L
        v += DT * p.force(x)
        if collide:
            acc += NU * npart * DT / 2.0
            nev, acc = n_events(acc)
            if nev:
                ke0 = 0.5 * float(np.dot(v, v))
                if arm in ("C3", "S-C3"):
                    pr, u = pairs_iid(npart, nev, rng)
                elif arm in ("C4a", "M4a"):
                    pr, u = pairs_vsort_cell(x, v, nev, rng)
                elif arm in ("C4b", "M4b", "S-C4b"):
                    pr, u = pairs_hilbert(x, v, nev, rng)
                else:
                    pr, u = pairs_hilbert(x, v, nev, rng, shuffle=True)
                if arm in ("M4a", "M4b"):
                    sg = rng.integers(0, 2, len(pr)) * 2 - 1
                    kac_rotate_monotone(v, pr, u, sg)
                else:
                    kac_rotate(v, pr, u)
                project_invariants(v, ke0)
        if k % SAMPLE_EVERY == 0:
            t = k * DT
            dj, dx = d_joint(x, v), d_k2(x)
            ts.append(t)
            djs.append(dj)
            dxs.append(dx)
            if writer:
                writer.write(f"{t},{dj:.10e},{dx:.10e}\n")
            if (arm.startswith("S-") and outdir
                    and any(abs(t - st) < 1e-9 for st in SNAP_TIMES)):
                _write_snapshot(outdir, arm, npart, offset, mode, t, x, v)
    if writer:
        writer.close()
    return np.array(ts), np.array(djs), np.array(dxs)


# ── Registered campaign drivers (Colab-ready, resumable) ─────────────────────
def run_registered_campaign(outdir, full=True, arms=R4_ARMS):
    """FULL r4: every registered (arm, offset, rung). r3 arms are locked
    out of campaign mode: offsets consumed, results of record."""
    assert set(arms) <= set(R4_ARMS), \
        f"campaign mode accepts r4 arms only {R4_ARMS}; r3 is of record"
    os.makedirs(outdir, exist_ok=True)
    mode = "FULL" if full else "QUICK"
    t_end = T_ECP if full else 5.0
    for arm in arms:
        ladder = ((LADDER_S if arm.startswith("S-") else LADDER_ECP)
                  if full else (2, 4))
        offs = OFFSETS[(arm, "ecp")] if full else (0,)
        for off in offs:
            t0 = time.time()
            for nppc in ladder:
                evolve(arm, "ecp", NG * nppc, off, t_end, mode, outdir)
            print(f"{arm} o{off} ladder done ({time.time()-t0:.0f}s)", flush=True)


def run_mcp_c2(outdir, full=True):
    """Deterministic C2-MCP companions (Nppc 1-3); consumes NO offsets."""
    os.makedirs(outdir, exist_ok=True)
    mode = "FULL" if full else "QUICK"
    for nppc in LADDER_MCP:
        evolve("C2", "mcp", NG * nppc, 0, T_MCP if full else 10.0, mode, outdir)
    print("C2-MCP companions done (deterministic; no offsets consumed)")


# ── Verification suite (r3: 8 registered + carried Hilbert exactness) ────────
def verify():
    print("=" * 66)
    print("HARNESS v1.2.0 VERIFICATION SUITE (r4; smoke offsets only)")
    print("=" * 66)
    t0 = time.time()

    gth = gate.theory_omega(8, 3, "tent", "mcp", 0.25)[0]
    gp = gate.pic_growth(8, 3, "tent", "mcp", 0.25, gth)
    rel = abs(gp - gth) / gth
    assert rel < 0.03, f"V0 FAIL: {rel:.3%}"
    print(f"V0 PASS  gate import regression: rel {rel:.1%} (record: 1.8%)")

    rng = rng_for(0)
    npart = 1024
    x, v = load_iid(npart, rng)
    v -= v.mean()
    ke_init = 0.5 * float(np.dot(v, v))
    acc, wp, wk = 0.0, 0.0, 0.0
    for _ in range(200):
        acc += NU * npart * DT / 2.0
        nev, acc = n_events(acc)
        ke0 = 0.5 * float(np.dot(v, v))
        pr, u = pairs_iid(npart, nev, rng)
        kac_rotate(v, pr, u)
        project_invariants(v, ke0)
        wp = max(wp, abs(float(v.sum())))
        wk = max(wk, abs(0.5 * float(np.dot(v, v)) - ke_init))
    assert wp < 1e-12 and wk < 1e-12, f"V1 FAIL: P={wp:.2e}, dKE={wk:.2e}"
    print(f"V1 PASS  Kac invariants over 200 steps: max|P|={wp:.1e}, "
          f"max|dKE|={wk:.1e} (<= 1e-12)")

    worst = max(d_k2(load_quiet(NG * n)[0]) for n in (2, 8, 32))
    assert worst < 1e-12, f"V2 FAIL: {worst:.2e}"
    print(f"V2 PASS  commensurate-lattice zero: max D = {worst:.1e}")

    npart, reps = 1024, 2560
    target = d_mc_joint(npart) ** 2
    tot = 0.0
    for off in SMOKE_OFFSETS:
        r = rng_for(off)
        for _ in range(reps // 2):
            tot += d_joint(r.random(npart), VTH * r.standard_normal(npart)) ** 2
    rel = abs(tot / reps - target) / target
    assert rel < 0.02, f"V3 FAIL: A1-J rel {rel:.3%}"
    print(f"V3 PASS  A1-J joint iid plateau: rel {rel:.2%} (< 2%; R={reps})")

    xr = (load_quiet(256)[0] + 0.3 * np.sin(2 * np.pi * np.arange(256) / 256)) % 1.0
    diff = abs(d_joint(xr, np.zeros(256), h=1e6) - d_k2(xr))
    assert diff < 1e-5, f"V4 FAIL: h-limit diff {diff:.2e}"
    print(f"V4 PASS  h->inf limit reproduces D_K2: |diff| = {diff:.1e}")

    n = 1024
    x0 = (np.arange(n) + 0.5) / n
    vq = VTH * ndtri((bit_reverse_perm(n) + 0.5) / n)
    worst = 0.0
    for t, rec in FS_ENVELOPE.items():
        r = d_joint((x0 + vq * t) % 1.0, vq) / d_mc_joint(n)
        worst = max(worst, abs(r - rec))
    assert worst < 6e-4, f"V5 FAIL: envelope dev {worst:.2e}"
    print(f"V5 PASS  free-streaming envelope regression: max dev = {worst:.1e}")

    for arm in ("C1", "C2", "C3", "C4a", "C4b", "CTRL",
                "M4a", "M4b", "S-C4b", "S-C3"):
        _, dj, dx = evolve(arm, "ecp", 128, 1, 2.0, "QUICK")
        assert np.all(np.isfinite(dj)) and np.all(np.isfinite(dx)), \
            f"V6 FAIL: {arm}"
    print("V6 PASS  all ten ECP arms execute (Kac + monotone + projections)")

    ts7, dj7, dx7 = evolve("C2", "mcp", 64, 0, 8.0, "QUICK")
    assert np.all(np.isfinite(dx7)) and dx7[0] < 1e-4 and dx7[-1] > 2 * dx7[0], \
        f"V7 FAIL: Dx0={dx7[0]:.2e}, DxT={dx7[-1]:.2e}"
    print(f"V7 PASS  C2-MCP companion: Dx {dx7[0]:.1e} -> {dx7[-1]:.1e} (growing)")

    from hilbertcurve.hilbertcurve import HilbertCurve
    rr = np.random.default_rng(1)
    pts = rr.integers(0, 256, size=(4096, 2))
    ref = np.asarray(HilbertCurve(p=8, n=2).distances_from_points(pts.tolist()))
    assert np.array_equal(hilbert_xy2d(pts[:, 0], pts[:, 1]), ref), "V8 FAIL"
    print("V8 PASS  vectorized Hilbert bit-exact vs reference (4096 pts)")

    nsamp = 1 << 20
    r9 = rng_for(1)
    va9 = VTH * r9.standard_normal(nsamp)
    vb9 = VTH * r9.standard_normal(nsamp)
    rad9 = np.hypot(va9, vb9)
    u9 = r9.random(nsamp)
    sg9 = r9.integers(0, 2, nsamp) * 2 - 1
    vam = -rad9 * np.cos(np.pi * u9)
    vbm = sg9 * rad9 * np.sin(np.pi * u9)
    ke_dev = float(np.max(np.abs((vam * vam + vbm * vbm)
                                 / (rad9 * rad9) - 1.0)))
    assert ke_dev < 1e-12, f"V9 FAIL: KE identity {ke_dev:.2e}"
    zs9 = np.sort(vam / rad9)
    ks9 = float(np.max(np.abs(np.arccos(-zs9) / np.pi
                              - (np.arange(1, nsamp + 1) - 0.5) / nsamp)))
    tol_ks = 2.5 / np.sqrt(nsamp)
    assert ks9 < tol_ks, f"V9 FAIL: KS {ks9:.2e} >= {tol_ks:.2e}"
    th9 = 2.0 * np.pi * r9.random(nsamp)
    var9 = va9 * np.cos(th9) + vb9 * np.sin(th9)
    vbr9 = -va9 * np.sin(th9) + vb9 * np.cos(th9)
    zmax = 0.0
    for fm, fr in ((vam * vbm, var9 * vbr9),
                   ((vam * vbm) ** 2, (var9 * vbr9) ** 2)):
        se = np.sqrt(fm.var() / nsamp + fr.var() / nsamp)
        zmax = max(zmax, abs(float(fm.mean() - fr.mean())) / se)
    assert zmax < 6.0, f"V9 FAIL: moment z {zmax:.2f} >= 6"
    ug = np.linspace(0.0, 1.0, 4097)
    assert np.all(np.diff(-np.cos(np.pi * ug)) > 0.0), "V9 FAIL: monotone"
    print(f"V9 PASS  monotone map exact-law: KE dev {ke_dev:.1e} (< 1e-12); "
          f"KS {ks9:.2e} (< {tol_ks:.2e}); moment z {zmax:.2f} (< 6); "
          f"monotone grid OK")

    print(f"\nAll 10 verification tests PASSED.  ({time.time()-t0:.1f}s)")
    print("=" * 66)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["verify", "campaign", "mcp_c2"],
                    default="verify")
    ap.add_argument("--outdir", default="/mnt/user-data/outputs")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--arms", default="M4a,M4b,S-C4b,S-C3")
    args = ap.parse_args()
    if args.mode == "verify":
        verify()
    elif args.mode == "mcp_c2":
        run_mcp_c2(args.outdir, full=args.full)
    else:
        run_registered_campaign(args.outdir, full=args.full,
                                arms=tuple(args.arms.split(",")))
