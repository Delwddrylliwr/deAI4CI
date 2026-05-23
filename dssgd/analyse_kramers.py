"""Diagnose the Kramers escape-time comparison."""
import sys, pickle, math, numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from analysis.stats import make_averaged_run, half_life, per_level_half_lives
from analysis.theory import effective_temperature, well_depth, kramers_escape_time

results_dir = Path('analysis_output/results')

def load_group(pattern):
    files = sorted(results_dir.glob(pattern))
    return [pickle.loads(f.read_bytes()) for f in files]

gamma = 1.0

print("=== KRAMERS DIAGNOSIS ===")
print("  (gamma=1.0, m=leaf_size=4, p=4.0)")
print()

for d in range(2, 6):
    runs = load_group(f'depth_sweep_trained__depth={d}__s*_r800.pkl')
    avg = make_averaged_run(runs, name=f'd={d}')
    T_eff = effective_temperature(avg.lr, avg.sigma2) if avg.sigma2 > 0 else 1.0
    print(f"d={d}  lr={avg.lr}  sigma2={avg.sigma2:.3f}  T_eff={T_eff:.5f}")

    print(f"  {'level':>6}  {'dV':>10}  {'dV/T_eff':>10}  {'Kramers_tau':>14}  {'obs_within_hl':>14}  {'obs_cross_hl':>13}")
    half_lives_within = per_level_half_lives(avg)

    # cross half-lives
    cross_hls = {}
    for lv in range(1, d + 1):
        cross_trace = avg.level_trace.cross_distances.get(lv, [])
        cross_hls[lv] = half_life(cross_trace)

    for lv in range(d + 1):
        dV = well_depth(gamma, avg.leaf_size, avg.p, lv) if lv > 0 else 0.0
        ratio = dV / T_eff if T_eff > 0 else float('inf')
        try:
            tau = kramers_escape_time(T_eff, dV)
        except OverflowError:
            tau = float('inf')
        tau_str = f'{tau:.1f}' if tau < 1e10 else f'~exp({ratio:.0f})'
        hl_w = half_lives_within.get(lv)
        hl_c = cross_hls.get(lv)
        print(f"  lv={lv:>3}  dV={dV:>8.4f}  ratio={ratio:>8.1f}  tau={tau_str:>14}  within_hl={str(hl_w):>5}  cross_hl={str(hl_c):>5}")
    print()

# Show what T_eff would make Kramers observable (tau ~ 100 rounds)
print("=== T_eff needed for observable Kramers times (tau < 100) ===")
print("  For dV = well_depth(gamma=1, leaf_size=4, p=4, level=lv):")
for lv in range(1, 6):
    dV = well_depth(1.0, 4, 4.0, lv)
    # tau < 100 => dV/T_eff < ln(100) => T_eff > dV/ln(100)
    T_needed = dV / math.log(100)
    lr_needed = T_needed * 2 / 2.0  # assuming sigma2=2
    print(f"  lv={lv}: dV={dV:.4f}  T_eff needed > {T_needed:.4f}  (requires lr > {lr_needed:.4f} @ sigma2=2)")
