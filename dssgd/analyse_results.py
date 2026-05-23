"""One-shot analysis script for analysis_output."""
import sys, pickle, numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from analysis.stats import (
    stationary_spread, make_averaged_run, level_coupling_ratios,
    level_stationary_variance, topological_dimension, convergence_rate, half_life
)
from analysis.theory import effective_temperature

results_dir = Path('analysis_output/results')

def load_group(pattern):
    files = sorted(results_dir.glob(pattern))
    return [pickle.loads(f.read_bytes()) for f in files]

# =====================================================================
# Temperature sweep (individual runs for spread)
# =====================================================================
print('=== TEMPERATURE SWEEP (stationary spread vs T_eff) ===')
print('  lr       T_eff     spread (mean+/-std)   sigma2')
for lr_str in ['0.001', '0.005', '0.01', '0.05', '0.1']:
    runs = load_group(f'temperature_sweep__lr={lr_str}__s*_r600.pkl')
    spreads = [stationary_spread(r) for r in runs]
    spreads = [s for s in spreads if s is not None]
    sigmas = float(np.mean([r.sigma2 for r in runs]))
    lr_val = runs[0].lr
    T = effective_temperature(lr_val, sigmas)
    if spreads:
        sp_str = f'{np.mean(spreads):.4f} +/- {np.std(spreads):.4f}'
    else:
        sp_str = 'N/A'
    print(f'  lr={lr_str}  T_eff={T:.4f}  spread={sp_str}  sigma2={sigmas:.4f}')

# =====================================================================
# Trained depth sweep - convergence and within-distance inspection
# =====================================================================
print()
print('=== TRAINED DEPTH SWEEP - global distance tail ===')
for d in range(2, 6):
    runs = load_group(f'depth_sweep_trained__depth={d}__s*_r800.pkl')
    r = runs[0]
    gd = r.level_trace.global_distances
    wd0 = r.level_trace.within_distances.get(0, [])
    gd_last = [f'{x:.2e}' for x in gd[-5:]] if gd else []
    wd_last = [f'{x:.2e}' for x in wd0[-5:]] if wd0 else []
    print(f'  d={d} s0: global[-5]={gd_last}')
    print(f'       within_l0[-5]={wd_last}')

# =====================================================================
# Convergence rate check on trained runs
# =====================================================================
print()
print('=== COUPLING RATES (trained, per-level log-linear fit) ===')
for d in range(2, 6):
    runs = load_group(f'depth_sweep_trained__depth={d}__s*_r800.pkl')
    avg = make_averaged_run(runs, name=f'trained_d={d}')
    rates = []
    for lv in range(d + 1):
        trace = avg.level_trace.within_distances.get(lv, [])
        rate = convergence_rate(trace)
        rates.append(rate)
    rate_strs = [f'{r:.4f}' if r is not None else 'N/A' for r in rates]
    print(f'  d={d}: rates_per_level={rate_strs}')
    ratios_raw = level_coupling_ratios(avg)
    ratio_strs = [f'{r:.2f}' if r is not None else 'N/A' for r in ratios_raw]
    print(f'       ratios={ratio_strs}  (theory SDE: ~2.0)')

# =====================================================================
# p-sweep: D vs p with depth-corrected pooling
# =====================================================================
print()
print('=== TOPOLOGICAL DIMENSION D vs p (p_sweep runs) ===')
for p in [1, 2, 4, 8, 16]:
    runs = load_group(f'p_sweep__p={p}__s*_r500.pkl')
    Ds = []
    for r in runs:
        nbhd = r.topology.mean_nbhd_by_radius
        D = topological_dimension(nbhd)
        if D is not None:
            Ds.append(D)
        print(f'    p={p} s: nbhd len={len(nbhd)}, D={D}')
    if Ds:
        print(f'  p={p}: D={np.mean(Ds):.3f} +/- {np.std(Ds):.3f}  (theory D~p)')
    else:
        print(f'  p={p}: no D computed')

# =====================================================================
# Spectral gap vs p
# =====================================================================
print()
print('=== SPECTRAL GAP vs p ===')
for p in [1, 2, 4, 8, 16]:
    runs = load_group(f'p_sweep__p={p}__s*_r500.pkl')
    gaps = [r.topology.spectral_gap for r in runs]
    print(f'  p={p}: spec_gap={np.mean(gaps):.4f} +/- {np.std(gaps):.4f}')
