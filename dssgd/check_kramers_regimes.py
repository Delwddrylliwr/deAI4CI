"""Check stationary variance ratios and cross-distance behaviour for Kramers sweep runs."""
import sys, pickle, numpy as np, math
sys.path.insert(0, '.')
from pathlib import Path
from analysis.stats import level_stationary_variance, make_averaged_run, time_to_half_stationary_cross
from analysis.theory import effective_temperature, well_depth

results_dir = Path('analysis_output/results')

def load_group(pattern):
    files = sorted(results_dir.glob(pattern))
    return [pickle.loads(f.read_bytes()) for f in files]

depth = 5
gamma, leaf_size, p = 1.0, 4, 4.0

print("=" * 70)
print("STATIONARY VARIANCE RATIOS — Kramers sweep runs (consensus init)")
print("  (theory: ratio per level = 2.0 when dV/T_eff >> 1)")
print("=" * 70)

for lr_str in ['0.01', '0.05', '0.1']:
    runs = load_group(f'kramers_sweep__lr={lr_str}__s*_r400.pkl')
    avg = make_averaged_run(runs, name=f'lr={lr_str}')
    T_eff = effective_temperature(avg.lr, avg.sigma2)
    variances = level_stationary_variance(avg, tail_fraction=0.3)
    levels = sorted(variances.keys())
    vals = [variances[lv] for lv in levels]
    ratios = [vals[i+1]/vals[i] for i in range(len(vals)-1) if vals[i] > 0]

    print(f"\nlr={lr_str}  T_eff={T_eff:.4f}  sigma2={avg.sigma2:.3f}")
    print(f"  {'level':>5}  {'dV':>7}  {'dV/T_eff':>9}  {'regime':>8}  {'variance':>12}")
    for lv in levels:
        dV = well_depth(gamma, leaf_size, p, lv) if lv > 0 else 0.0
        ratio_to_T = dV / T_eff if T_eff > 0 else float('inf')
        regime = "deep" if ratio_to_T > 3 else ("shallow" if ratio_to_T < 1 else "medium")
        print(f"  lv={lv:>3}  dV={dV:>5.3f}  ratio={ratio_to_T:>7.2f}  {regime:>8}  var={variances.get(lv, float('nan')):.3e}")

    print(f"  variance ratios (var[l+1]/var[l]): {[f'{r:.2f}' for r in ratios]}")
    print(f"  (deep-well theory predicts 2.0 for each)")

print()
print("=" * 70)
print("ΔV/T_eff PER LEVEL — all runs, to map coverage")
print("=" * 70)
print(f"  {'lr':>6}  {'T_eff':>7}  " + "  ".join(f"lv{lv}" for lv in range(1, depth+1)))
for lr_str in ['0.001', '0.005', '0.01', '0.05', '0.1']:
    # Use temperature_sweep sigma2 for this lr if available, else kramers
    try:
        runs = load_group(f'kramers_sweep__lr={lr_str}__s*_r400.pkl')
    except Exception:
        runs = load_group(f'temperature_sweep__lr={lr_str}__s*_r600.pkl')
    if not runs:
        continue
    avg = make_averaged_run(runs)
    T_eff = effective_temperature(avg.lr, avg.sigma2)
    ratios_str = []
    for lv in range(1, depth+1):
        dV = well_depth(gamma, leaf_size, p, lv)
        r = dV / T_eff
        ratios_str.append(f"{r:>5.1f}")
    print(f"  {lr_str:>6}  {T_eff:>7.4f}  " + "  ".join(ratios_str))

print()
print("Rows of interest from Kramers table and where they fall:")
print("  Stationary variance regime = dV/T_eff > 3 (deep, theory predicts 2x ratio)")
print("  Kramers-testable regime    = dV/T_eff < 5 (tau < ~150 rounds)")
print("  These regimes OVERLAP at dV/T_eff ~ 3-5 (e.g. lr=0.1 level 3, lr=0.01 level 4)")
