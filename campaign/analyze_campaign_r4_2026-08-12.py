"""Registered analysis — Session 19 / r4 campaign (monotone coupling).

Implements EXACTLY the r4 registration (session_log 2026-08-12): M4a/M4b
window-means over W* = [20,40], per-replicate ladder slopes, 95% log-space
t CIs, r3-identical prediction bands and falsifiers A/B; the secondary
mechanism-directional contrast WM(M4x) < WM(C4x-of-record) at every rung
Np >= 512 (95% Welch CI on log WMs, rung passes iff CI_high(ratio) < 1);
the S-arm D_v snapshot criterion (Gaussian kernel h_v = v_th/2, analytic
iid plateau sqrt((2/3)/Np), window-mean = geometric mean over snapshot
times {20, 40}; S-C4b ratio >= 2 at both Np, S-C3 within +/-25%).
Estimator choices inherited from analyze_campaign_2026-08-07.py; choices
where the registration is silent are stated in-line and printed.

Inputs: r4 results dir, r3 results dir (frozen curves), outdir.
Outputs: verdicts (stdout), r4_campaign_summary_2026-08-12.csv,
r4_dv_summary_2026-08-12.csv, r4_ladder_2026-08-12.png.

Author: James M. Hyman (Tulane) with Claude — Session 19. 2026-08-12.
"""

import csv
import math
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R4 = sys.argv[1] if len(sys.argv) > 1 else "r4res"
R3 = sys.argv[2] if len(sys.argv) > 2 else "r3res"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/mnt/user-data/outputs"

NG, VTH = 64, 0.05
DG, HV = 1.0 / NG, 0.05 / 2.0
LAD = (128, 256, 512, 1024, 2048, 4096, 8192)
LAD_S = (2048, 8192)
LAD_SEC = (512, 1024, 2048, 4096, 8192)
ARMS_M = {"M4a": range(62, 68), "M4b": range(68, 74)}
ARMS_S = {"S-C4b": range(74, 77), "S-C3": range(77, 80)}
ARMS_C4 = {"C4a": range(38, 44), "C4b": range(44, 50)}
PAIR = {"M4a": "C4a", "M4b": "C4b"}
SNAP_T = (0, 10, 20, 40)


def d_mc_joint(n):
    return math.sqrt((2.0 / (3 * DG) - HV / math.sqrt(HV ** 2 + 2 * VTH ** 2)) / n)


def d_mc_v(n):
    """Analytic iid plateau for D_v: sqrt((1 - h/sqrt(h^2+2s^2))/Np) = sqrt((2/3)/Np)."""
    return math.sqrt((1.0 - HV / math.sqrt(HV ** 2 + 2 * VTH ** 2)) / n)


def load(res, arm, npart, off):
    fn = os.path.join(res, f"discrepancy_{arm}_ecp_Np{npart}_o{off}_FULL.csv")
    a = np.genfromtxt(fn, delimiter=",", names=True)
    return a["t"], a["DJ"], a["Dx"]


def tci_log(vals):
    lv = np.log(np.asarray(vals, dtype=float))
    n, m = len(lv), lv.mean()
    if n < 2:
        return math.exp(m), float("nan"), float("nan")
    half = stats.t.ppf(0.975, n - 1) * lv.std(ddof=1) / math.sqrt(n)
    return math.exp(m), math.exp(m - half), math.exp(m + half)


def tci_lin(vals):
    v = np.asarray(vals, dtype=float)
    n, m = len(v), v.mean()
    if n < 2:
        return m, float("nan"), float("nan")
    half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return m, m - half, m + half


def d_v(v, h=HV, block=1024):
    """Velocity-marginal kernel discrepancy vs N(0, VTH^2); v-factor of D_J."""
    n = v.size
    t1 = 0.0
    for i0 in range(0, n, block):
        dv = v[i0:i0 + block, None] - v[None, :]
        t1 += float(np.exp(-dv * dv / (2 * h * h)).sum())
    t1 /= n * n
    g = (h / np.hypot(h, VTH)) * np.exp(-v * v / (2 * (h * h + VTH * VTH)))
    t3 = h / math.sqrt(h * h + 2 * VTH * VTH)
    return float(math.sqrt(max(t1 - 2.0 * float(g.mean()) + t3, 0.0)))


def wm_of(res, arm, offs, lad):
    out = {}
    for off in offs:
        for n in lad:
            t, dj, _ = load(res, arm, n, off)
            w = (t >= 20) & (t <= 40)
            out[(off, n)] = dj[w].mean()
    return out


# ── frozen r3 curves (recomputed with the identical WM code path) ────────────
WM_C4 = {arm: wm_of(R3, arm, offs, LAD) for arm, offs in ARMS_C4.items()}
r3sum = {}
with open(os.path.join(OUT, "campaign_summary_2026-08-07.csv")) as f:
    for row in csv.DictReader(f):
        r3sum[(row["arm"], int(row["Np"]))] = float(row["geo_mean_WM_DJ"])
worst = 0.0
for arm, offs in ARMS_C4.items():
    for n in LAD:
        g = tci_log([WM_C4[arm][(o, n)] for o in offs])[0]
        worst = max(worst, abs(g / r3sum[(arm, n)] - 1.0))
assert worst < 1e-4, f"frozen-curve cross-check failed: {worst:.2e}"
print(f"frozen r3 C4 curves recomputed; cross-check vs summary of record: "
      f"max rel dev {worst:.1e} (PASS)")

# ── M-arms: registered verdicts (r3-identical logic) ─────────────────────────
WM_M = {arm: wm_of(R4, arm, offs, LAD) for arm, offs in ARMS_M.items()}
summary_rows = []
print("=" * 78)
print("r4 REGISTERED VERDICTS — W* = [20,40], D_MC_J = 6.5064/sqrt(Np)")
print("=" * 78)
hdr = "Np:    " + "".join(f"{n:>8d}" for n in LAD)
primary = {}
for arm, offs in ARMS_M.items():
    ratios, lows, highs = [], [], []
    for n in LAD:
        g, lo, hi = tci_log([WM_M[arm][(o, n)] for o in offs])
        ratios.append(g / d_mc_joint(n))
        lows.append(lo / d_mc_joint(n))
        highs.append(hi / d_mc_joint(n))
        summary_rows.append([arm, n, len(offs), f"{g:.6e}",
                             f"{g / d_mc_joint(n):.4f}",
                             f"{lo / d_mc_joint(n):.4f}",
                             f"{hi / d_mc_joint(n):.4f}"])
    sl = [np.polyfit(np.log(LAD),
                     np.log([WM_M[arm][(o, n)] for n in LAD]), 1)[0]
          for o in offs]
    m, lo, hi = tci_lin(sl)
    print(f"\n{arm} ({len(offs)} replicates)")
    print(hdr)
    print("ratio: " + "".join(f"{r:8.3f}" for r in ratios))
    print("CIlow: " + "".join(f"{r:8.3f}" for r in lows))
    print("CIhi:  " + "".join(f"{r:8.3f}" for r in highs))
    print(f"slope: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    pred_wm = all(r <= 0.5 for r in ratios)
    pred_sl = hi <= -0.65 and m >= -1.05
    fA = sum(l >= 0.8 for l in lows)
    fB = lo >= -0.55
    primary[arm] = dict(pred_wm=pred_wm, pred_sl=pred_sl, fA=fA, fB=fB,
                        slope=(m, lo, hi), ratios=ratios)
    print(f"PREDICTION {arm}: WM<=0.5 at all rungs: "
          f"{'MET' if pred_wm else 'NOT MET'}; slope in [-1.05,-0.65]: "
          f"{'MET' if pred_sl else 'NOT MET'} (point pred -0.85)")
    print(f"FALSIFIER {arm}: rungs with CIlow>=0.8: {fA} (trip at >=2) -> "
          f"{'TRIP' if fA >= 2 else 'no'}; slope CIlow>=-0.55 -> "
          f"{'TRIP' if fB else 'no'}  ==> SUB-ARM "
          f"{'FAILS' if (fA >= 2 or fB) else 'stands'}")

# ── secondary mechanism-directional contrast ─────────────────────────────────
print("\n" + "=" * 78)
print("SECONDARY CONTRAST — WM(M4x) < WM(C4x of record), rungs Np >= 512")
print("(95% Welch CI on log WMs; rung passes iff CI_high(ratio) < 1)")
print("=" * 78)
secondary = {}
for arm in ARMS_M:
    carm = PAIR[arm]
    ok_all = True
    print(f"\n{arm} vs {carm}:")
    for n in LAD_SEC:
        lm = np.log([WM_M[arm][(o, n)] for o in ARMS_M[arm]])
        lc = np.log([WM_C4[carm][(o, n)] for o in ARMS_C4[carm]])
        d = lm.mean() - lc.mean()
        se = math.sqrt(lm.var(ddof=1) / len(lm) + lc.var(ddof=1) / len(lc))
        df = (lm.var(ddof=1) / len(lm) + lc.var(ddof=1) / len(lc)) ** 2 / (
            (lm.var(ddof=1) / len(lm)) ** 2 / (len(lm) - 1)
            + (lc.var(ddof=1) / len(lc)) ** 2 / (len(lc) - 1))
        half = stats.t.ppf(0.975, df) * se
        rlo, r, rhi = math.exp(d - half), math.exp(d), math.exp(d + half)
        ok = rhi < 1.0
        ok_all &= ok
        print(f"  Np={n:>5d}: ratio M/C4 = {r:.4f}  "
              f"95% CI [{rlo:.4f}, {rhi:.4f}]  -> {'PASS' if ok else 'FAIL'}")
    secondary[arm] = ok_all
    print(f"  SECONDARY {arm}: {'HOLDS at every rung' if ok_all else 'FAILS'}")

# ── S-arms: D_v snapshot criterion ───────────────────────────────────────────
print("\n" + "=" * 78)
print("S-ARMS — D_v snapshot criterion (analytic plateau sqrt((2/3)/Np);")
print("window-mean = geometric mean over snapshot times {20, 40})")
print("=" * 78)
rng = np.random.default_rng(0)
chk = np.mean([d_v(VTH * rng.standard_normal(2048)) / d_mc_v(2048)
               for _ in range(8)])
print(f"plateau sanity (8 iid draws, Np=2048): mean ratio {chk:.3f} (~1)")
dv_rows = []
dv_verdict = {}
for arm, offs in ARMS_S.items():
    print(f"\n{arm}:")
    per_np = {}
    for n in LAD_S:
        wms = []
        for off in offs:
            r_t = {}
            for tt in SNAP_T:
                z = np.load(os.path.join(
                    R4, f"snapshot_{arm}_Np{n}_o{off}_t{tt}.npz"))
                r_t[tt] = d_v(z["v"]) / d_mc_v(n)
            wm = math.sqrt(r_t[20] * r_t[40])
            wms.append(wm)
            dv_rows.append([arm, n, off, f"{r_t[0]:.4f}", f"{r_t[10]:.4f}",
                            f"{r_t[20]:.4f}", f"{r_t[40]:.4f}", f"{wm:.4f}"])
        g, lo, hi = tci_log(wms)
        per_np[n] = g
        print(f"  Np={n:>5d}: WM D_v ratio geo-mean {g:.3f} "
              f"[{lo:.3f}, {hi:.3f}] (per-offset {min(wms):.3f}-{max(wms):.3f})")
    if arm == "S-C4b":
        ok = all(per_np[n] >= 2.0 for n in LAD_S)
        dv_verdict[arm] = ok
        print(f"  CRITERION S-C4b (>=2 at both Np): {'MET' if ok else 'NOT MET'}")
    else:
        ok = all(abs(per_np[n] - 1.0) <= 0.25 for n in LAD_S)
        dv_verdict[arm] = ok
        print(f"  CRITERION S-C3 (within +/-25%): {'MET' if ok else 'NOT MET'}")

# ── registered outcome ───────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("REGISTERED OUTCOME")
print("=" * 78)
for arm in ARMS_M:
    p = primary[arm]
    stands = not (p["fA"] >= 2 or p["fB"])
    full = p["pred_wm"] and p["pred_sl"]
    print(f"{arm}: PRIMARY {'MET (full band recovery)' if full else 'NOT MET'}"
          f" | falsifiers: {'clean' if stands else 'TRIPPED'}"
          f" | SECONDARY vs {PAIR[arm]}: "
          f"{'holds' if secondary[arm] else 'fails'}")
print(f"S-C4b mechanism criterion: "
      f"{'MET' if dv_verdict.get('S-C4b') else 'NOT MET'}; "
      f"S-C3 control: {'clean' if dv_verdict.get('S-C3') else 'OUT OF BAND'}")

# ── artifacts ────────────────────────────────────────────────────────────────
with open(os.path.join(OUT, "r4_campaign_summary_2026-08-12.csv"), "w",
          newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "Np", "n_rep", "geo_mean_WM_DJ", "ratio_to_DMCJ",
                "CI95_low_ratio", "CI95_high_ratio"])
    w.writerows(summary_rows)
with open(os.path.join(OUT, "r4_dv_summary_2026-08-12.csv"), "w",
          newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "Np", "offset", "Dv_ratio_t0", "Dv_ratio_t10",
                "Dv_ratio_t20", "Dv_ratio_t40", "WM_Dv_ratio"])
    w.writerows(dv_rows)

fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
for arm in ("C1", "C3", "C4a", "C4b", "CTRL"):
    g = [r3sum[(arm, n)] for n in LAD]
    ax.loglog(LAD, g, lw=0.9, ms=3, alpha=0.55, marker=".",
              label=f"{arm} (r3, frozen)")
mk = {"M4a": "D", "M4b": "v"}
for arm, offs in ARMS_M.items():
    g = [tci_log([WM_M[arm][(o, n)] for o in offs])[0] for n in LAD]
    ax.loglog(LAD, g, marker=mk[arm], label=f"{arm} (r4)", lw=1.6, ms=6)
npf = np.array(LAD, dtype=float)
ax.loglog(npf, [d_mc_joint(n) for n in npf], "k--", lw=1, label=r"$D_{MC,J}$")
ax.loglog(npf, [0.5 * d_mc_joint(n) for n in npf], "k:", lw=1,
          label=r"$0.5\,D_{MC,J}$")
ax.set_xlabel(r"$N_p$")
ax.set_ylabel(r"window-mean $D_J$ over $t\in[20,40]$")
ax.set_title("r4 campaign ladder: monotone coupling vs frozen r3")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r4_ladder_2026-08-12.png"))
print(f"\nartifacts: r4_campaign_summary_2026-08-12.csv, "
      f"r4_dv_summary_2026-08-12.csv, r4_ladder_2026-08-12.png -> {OUT}")
