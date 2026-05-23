"""Show how the observable Kramers window shifts with leaf_size."""
import sys, math
sys.path.insert(0, '.')
from analysis.theory import well_depth, effective_temperature, kramers_escape_time

gamma, branching, depth, p = 1.0, 2, 5, 4.0
lr, sigma2_approx = 0.1, 1.7   # approx T_eff at lr=0.1

T_eff = effective_temperature(lr, sigma2_approx)
print(f"lr={lr}, T_eff approx {T_eff:.4f}\n")
print(f"{'m':>4}  {'n_agents':>9}  " + "  ".join(f"lv{lv}:dV/T" for lv in range(1,6)))

for m in [4, 8, 16, 32]:
    n = branching**depth * m
    cols = []
    for lv in range(1, depth+1):
        dV = well_depth(gamma, m, p, lv)
        ratio = dV / T_eff
        try:
            tau = kramers_escape_time(T_eff, dV)
            tau_str = f"{ratio:.1f}({tau:.0f}r)" if tau < 1e4 else f"{ratio:.1f}(>>)"
        except OverflowError:
            tau_str = f"{ratio:.1f}(>>)"
        cols.append(tau_str)
    print(f"  m={m:>2}  n={n:>6}  " + "  ".join(f"{c:>14}" for c in cols))

print()
print("Format: dV/T_eff(tau_rounds). '>>' means unobservable (tau > 1e4).")
print("Observable window = levels where tau < ~400 rounds (our budget).")
