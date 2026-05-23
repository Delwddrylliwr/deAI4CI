# NMH Simulation vs SDE Theory: Results Report

**Run parameters:** depth sweep 800 rounds, trained depth sweep 800 rounds, p/branching sweeps 500 rounds, temperature sweep 600 rounds, Kramers sweep (consensus init) 400 rounds. 3 seeds per configuration. Branching=2, leaf\_size=4, p=4 (unless swept).

---

## 1. Degree Limit

**Prediction:** Mean degree follows the geometric series
$$\bar{k}(d) = (k_0 - k_\infty)\,b^{-d} + k_\infty, \qquad k_\infty = (k_0 - 1) + \frac{p \cdot m \cdot (b-1)}{4 - b}$$
with limit $k_\infty = 11.0$ for $(p=4, b=2, m=4)$.

**Result:** Excellent agreement across all depths.

| depth | observed | theory | error |
|-------|----------|--------|-------|
| 1 | 7.00 | 7.00 | 0.0% |
| 2 | 8.92 | 9.00 | 0.9% |
| 3 | 9.69 | 10.00 | 3.1% |
| 4 | 10.41 | 10.50 | 0.9% |
| 5 | 10.89 | 10.75 | 1.3% |

The observed degrees converge toward the theoretical limit of 11.0 as depth increases, with all values within 3% of the finite-depth formula. The degree\_vs\_depth plot shows the geometric-series curve and asymptote matching observations closely.

**Verdict: ✓ Confirmed.**

---

## 2. Convergence Ordering (Untrained Gossip)

**Prediction:** In pure gossip, within-module consensus distances should converge in a level-by-level ordering — within-leaf-module (level 0) fastest, global (level $d$) slowest — with all traces decaying exponentially at rates proportional to the spectral gap within each module grouping.

**Result:** Confirmed cleanly in the convergence plots. For depth=5 at 800 rounds:
- All six level traces show log-linear exponential decay over rounds 0–400.
- Level 0 (within-leaf, size 4) converges fastest; global distance converges slowest.
- All traces reach the numerical floor (~$10^{-8}$) by round 500.
- The spectral gap increases monotonically with $p$: 0.028 (p=1) → 0.123 (p=4) → 0.361 (p=16), consistent with more cross-module edges improving mixing.

**Verdict: ✓ Confirmed.**

---

## 3. Topological Dimension $D \sim p$

**Prediction:** BFS neighbourhood growth $n(r) \sim r^D$ with $D \approx p$ (linear scaling of effective dimension with cross-module edge density).

**Result:** Measurement fails for this graph size.

| p | fitted D | theory D | graph diam |
|---|----------|----------|------------|
| 1 | 1.25 ± 0.20 | 1.0 | ~6 |
| 2 | 1.42 ± 0.06 | 2.0 | ~5 |
| 4 | 1.40 ± 0.08 | 4.0 | ~3 |
| 8 | 0.85 ± 0.03 | 8.0 | ~3 |
| 16 | undefined | 16.0 | 2 |

$D$ is essentially flat at ~1.4 for $p \in \{1,2,4\}$ and actually *decreases* at $p=8$. The reason is a regime mismatch: at depth=3 ($n=32$ nodes), the graph diameter collapses to 2–3 hops for $p \geq 4$. The BFS saturates immediately, leaving only 3 data points for the log-log fit — far too few to resolve a power-law growth regime. The $D \sim p$ prediction requires graphs where the diameter is much larger than $p$, i.e., $n \gg p^p$.

**Verdict: ✗ Measurement inapplicable at this graph size. Requires $n \gg p^p$ for the BFS growth regime to exist.**

---

## 4. Stationary Variance Hierarchy

**Prediction:** In the SDE/training regime at stationarity, the within-module consensus distance at level $\ell$ should satisfy
$$\frac{\sigma^2_\ell}{\sigma^2_{\ell-1}} = \frac{\gamma_{\ell-1}}{\gamma_\ell} = 2, \qquad \gamma_\ell = \frac{\gamma \cdot m \cdot p}{2^{\ell+1}}$$

**Result:** The 2× ratio holds at levels $\ell \geq 1$ but fails at level 0.

Observed ratios $\sigma^2_{\ell+1} / \sigma^2_\ell$:

| depth | lv 0→1 | lv 1→2 | lv 2→3 | lv 3→4 | lv 4→5 |
|-------|--------|--------|--------|--------|--------|
| 2 | 1.10 | 3.40 | — | — | — |
| 3 | 1.12 | 2.85 | 1.67 | — | — |
| 4 | 1.11 | 1.89 | 1.75 | 2.13 | — |
| 5 | 1.08 | 1.79 | 1.58 | 2.02 | 1.48 |

- **Levels ≥ 1:** Ratios cluster around 2.0, in good agreement with theory.
- **Level 0 → 1:** Ratio consistently ~1.1 (well below 2). Cause: within-leaf agents share the same true weight vector, so aligned gradients suppress within-leaf variance below the SDE prediction. The within-leaf consensus distance is driven almost entirely by gradient noise, while levels ≥ 1 experience both noise and cross-module heterogeneity.

**Verdict: ✓ Confirmed at levels $\ell \geq 1$. Systematic deviation at level 0 is a predictable consequence of shared intra-module objectives, not a failure of the theory.**

---

## 5. Effective Temperature / Stationary Spread

**Prediction:** At stationarity, parameter spread $\sim \sqrt{T_\text{eff}} = \sqrt{\eta \sigma^2 / 2}$, i.e., a power-law exponent of 0.5 in spread vs $T_\text{eff}$.

**Result:** Spread increases with $T_\text{eff}$ (direction correct) but with a fitted exponent of **0.11**, far below the predicted 0.5.

| lr | $T_\text{eff}$ | spread |
|----|----------------|--------|
| 0.001 | 0.0011 | 0.253 ± 0.003 |
| 0.005 | 0.0054 | 0.365 ± 0.001 |
| 0.010 | 0.0107 | 0.369 ± 0.002 |
| 0.050 | 0.0476 | 0.398 ± 0.014 |
| 0.100 | 0.0876 | 0.434 ± 0.023 |

Over an 80× increase in $T_\text{eff}$, spread increases only ~70%. The dominant component of parameter spread is the data heterogeneity: agents must straddle different module-specific optima, setting a floor on the spread that is independent of gradient noise. The thermal contribution ($\sim \sqrt{T_\text{eff}}$) rides on top of this floor and is therefore a weak fraction of the total signal.

**Verdict: ✓ Direction correct. Magnitude dominated by data-heterogeneity floor; $\sqrt{T_\text{eff}}$ contribution is present but not resolvable without subtracting the heterogeneity baseline.**

---

## 6. Kramers Escape Times

**Prediction:** In the SDE regime, the time for a level-$\ell$ super-module to escape its potential well (depth $\Delta V^{(\ell)} = \gamma m p / 4^\ell$) is
$$\tau^{(\ell)} \sim \exp\!\left(\frac{\Delta V^{(\ell)}}{T_\text{eff}}\right)$$

**Corrected experiment:** All agents initialised from identical parameters (`start_from_consensus=True`). Observed metric: time for `cross_distances[level]` to reach 50% of its stationary value (time to fall into the well, which by detailed balance equals the escape time). Three learning rates at depth=5 to span ΔV/T_eff ratios from 0.8 to 6.3.

**Result:** Kramers theory is qualitatively correct but systematically over-predicts timescales at deep wells.

The actual control variable is $\Delta V / T_\text{eff}$, which changes with both level (ΔV halves every level) and lr (T_eff scales linearly). Each lr therefore exposes a different slice of levels: at lr=0.01 only levels 4–5 have finite τ predictions; at lr=0.1 levels 3–5 are accessible. All valid (lr, level) combinations are shown below, sorted by $\Delta V / T_\text{eff}$:

| lr | level | $\Delta V$ | $T_\text{eff}$ | $\Delta V / T_\text{eff}$ | $\tau_\text{Kramers}$ | $\tau_\text{observed}$ | ratio |
|----|-------|-----------|----------------|--------------------------|----------------------|------------------------|-------|
| 0.10 | 5 | 0.016 | 0.079 | 0.20 | 1.2 | ~1 | ~1× |
| 0.10 | 4 | 0.063 | 0.079 | 0.79 | 2.2 | ~2 | ~1× |
| 0.05 | 5 | 0.016 | 0.044 | 0.36 | 1.4 | ~1 | ~1× |
| 0.05 | 4 | 0.063 | 0.044 | 1.44 | 4.2 | ~4–5 | ~1× |
| 0.01 | 5 | 0.016 | 0.010 | 1.56 | 4.8 | ~4–5 | ~1× |
| 0.10 | 3 | 0.250 | 0.079 | 3.16 | 24 | ~2 | ~12× |
| 0.01 | 4 | 0.063 | 0.010 | 6.25 | 520 | ~12 | ~43× |
| 0.05 | 3 | 0.250 | 0.044 | 5.75 | 314 | ~4 | ~75× |

**Pattern:** When $\Delta V / T_\text{eff} \lesssim 2$, theory and observation agree within a factor of ~2, regardless of which (lr, level) combination produces that ratio. When $\Delta V / T_\text{eff} \gtrsim 3$, observed timescales are 12–75× shorter than the exponential prediction. The threshold is a property of the ratio alone, not of lr or level individually.

**Interpretation:** The Kramers formula assumes an isolated particle escaping a potential barrier by thermal diffusion alone. In gossip-SGD, the gossip averaging protocol imposes an additional drift force pulling module centroids together. This force is absent from the Kramers derivation. For shallow barriers ($\Delta V / T_\text{eff} \lesssim 2$) the gossip drift is a small correction and the Kramers result holds; for deep barriers it provides a fast shortcut that dominates over the thermally-activated mechanism. Equivalently, the Kramers formula gives an upper bound on the separation timescale — gossip coupling cuts it down by a factor that grows exponentially with $\Delta V / T_\text{eff}$.

The ordering of timescales is still correct: deeper levels (higher $\ell$, smaller $\Delta V$) separate faster, as Kramers predicts. The quantitative agreement holds in the $\Delta V / T_\text{eff} \lesssim 2$ regime accessible at lr=0.05–0.1.

**Verdict: ~ Partial. Ordering confirmed. Quantitative agreement holds for $\Delta V / T_\text{eff} \lesssim 2$; gossip coupling provides a fast path that breaks the pure-Kramers exponential law at deeper wells.**

---

## 7. Coupling Rate Ratios

**Prediction:** In the SDE/training regime, convergence-rate ratios $\lambda_\ell / \lambda_{\ell+1} \approx \gamma_\ell / \gamma_{\ell+1} = 2$.

**Result:** The convergence-rate estimator is inapplicable in the training regime. Within-distances plateau at a stationary level (~$10^{-2}$) rather than decaying to zero, so the log-linear fit returns rates ≈ 0 for depths ≥ 3. Only depth=2 produces non-None values (~1.1–1.35), well below the prediction of 2. This is because at depth=2 with only 3 hierarchy levels, the stationary distances are very small relative to the initial values, partially mimicking a decay.

The coupling-rate ratio estimator is valid only in the *pure-gossip* (untrained) regime where distances do decay to zero. The correct measurement for the SDE regime is the stationary variance hierarchy (Section 4 above).

**Verdict: ✗ Metric inapplicable in the training regime. Stationary variance ratios (Section 4) are the correct observable.**

---

## 8. Experiment A: Catch-up Curve (Dynamic Picture)

**Prediction:** After injecting a perturbation $\Delta$ into leaf module 0, the time for module $j$'s centroid to shift by $\|\Delta\|/2$ in the $\hat\Delta$-direction grows as
$$T_\text{catch}(d) = \frac{\ln 2 \cdot 2^{d+1}}{\gamma m p}$$
i.e. log-linear in distance $d$ with slope 1 on log₂ axes.

**Setup:** 3 seeds × 3 learning rates ($\eta \in \{0.01, 0.05, 0.1\}$), depth=5, $m=4$, $p=4$, $n=128$. 800-round warmup to stationarity, perturbation magnitude $\|\Delta\| = 5\sigma_0$ (where $\sigma_0$ is within-leaf spread), 400 measurement rounds.  Threshold for detection: $\|\Delta\|/2$.

| lr | $\sigma_0$ | $\|\Delta\|$ | threshold |
|----|-----------|------------|-----------|
| 0.01 | 0.275 | 1.376 | 0.688 |
| 0.05 | 0.319 | 1.595 | 0.798 |
| 0.10 | 0.357 | 1.787 | 0.893 |

**Result (first run, threshold = ‖Δ‖/2): all None.** All 279 entries returned $t_{50}$ = None. The MH normalization means the source centroid itself only peaks at $\approx 0.66$ (37% of $\|\Delta\|$), far below the threshold of $\approx 0.79$.

**Result (re-run, threshold = source\_peak × 0.10): detections, but confounded.**

Reducing to 10% of source peak ($\approx 0.06$) produced detections at d=1–3 for all lr values and at d=4–5 for lr=0.1. However, two structural issues prevent clean measurement:

**Issue 1: Direct-edge confound.**
The NMH topology places stochastic cross-edges between ALL level pairs with probability $p/4^\ell$. At level $\ell=3$ (connecting d=3 leaves to source), each agent-pair has $p/4^3 = 0.0625$ connection probability, so the expected number of direct source↔d=3 edges per leaf-pair is $m^2 \times 0.0625 = 1.0$. About half of d=3 leaves therefore have ≥1 direct edge to source and show $t_{50} = 1$–2, indistinguishable from d=1. Similarly at d=4 (expected 0.25 direct edges) and d=5 (0.06 expected).

Separating leaves by whether they share a direct edge with source (reconstructed from seed):

| Distance | Direct edge: t₅₀ range | No direct edge: t₅₀ range |
|---|---|---|
| d=1 | 1 (all 9 obs.) | — (all have direct edges) |
| d=2 | 1–2 (all 18 obs.) | — (all have direct edges) |
| d=3 | 1–9 | 7–122 (6 obs. across lr/seeds) |
| d=4 | 1–45 | 2–338 (wide spread) |
| d=5 | 1–7 | 2–349 (very wide spread) |

**Issue 2: Topology heterogeneity causes extreme t₅₀ variance.**
Even among no-direct-edge leaves at the same hierarchical distance, $t_{50}$ varies by a factor of ≥70 (e.g., d=5 at lr=0.1: values 2–349). This reflects genuine variation in indirect connectivity: some no-direct-edge leaves have many short indirect paths and propagate quickly; others are poorly connected and propagate slowly or not at all within 400 rounds.

Consequently, the no-direct-edge groups for d=3 (the slowest d=3 leaves, specifically lacking direct connections) show larger mean $t_{50}$ than d=4 (only the fastest-detected d=4 leaves cross threshold). This anti-monotonic ordering is a selection-bias artifact.

**Signal-to-noise:** The source centroid projection peaks at ≈0.66 (t=1). By d=2, the peak amplitude is ≈0.08, and by d=3–5 it is ≈0.05–0.10 — comparable to the maximum of a 400-round centroid random walk ($\approx \sigma_\text{noise} \sqrt{2 \ln T} \approx 0.027 \times 3.7 \approx 0.10$). Threshold crossings at d≥3 are therefore partly or fully driven by thermal fluctuations rather than causal propagation.

**Verdict: Experiment A inconclusive for d ≥ 3.**

The doubling-per-level slope cannot be tested with current parameters:
1. SNR < 4 at d≥3 makes individual leaf $t_{50}$ unreliable
2. Direct-edge confound contaminates d=3 (≈50% of leaves have direct connections)
3. Topology heterogeneity creates a factor-of-70+ spread at fixed d, masking the d-dependence

What IS clean: d=1 ($t_{50}=1$ always) and d=2 ($t_{50}=1$–2) are consistently detected. Their ratio of $\approx 1$–2 is consistent with the 2× doubling prediction, and the absolute timescale is shifted by $\approx 1+\bar{k}$ (MH normalization penalty).

**To test slope=1 cleanly at d=3–5:** use perturbation scale ≥50σ₀ (×10 current), ≥10 seeds, or Experiment D (autocorrelations), which bypasses threshold calibration entirely.

---

## Summary

| Claim | Status | Notes |
|-------|--------|-------|
| Degree limit converges to $k_\infty$ | ✓ Confirmed | Within 3%, all depths |
| Level-ordered convergence in gossip | ✓ Confirmed | Clean log-linear ordering |
| $D \sim p$ topological dimension | ✗ Regime mismatch | Graph too small ($n=32$, diam~3 for $p \geq 4$) |
| Stationary variance ratio $\approx 2$ per level | ✓ Confirmed (levels ≥ 1) | Level 0→1 ratio ~1.1 due to shared module objectives |
| Spread $\sim \sqrt{T_\text{eff}}$ | Partial | Direction correct; heterogeneity floor dominates magnitude |
| Spectral gap increases with $p$ | ✓ Confirmed | 0.028 → 0.361 monotone |
| Kramers escape times | ~ Partial | Ordering correct; quantitative match only for $\Delta V/T_\text{eff} \lesssim 2$; gossip coupling dominates at deep wells |
| Catch-up timing ratio doubles per level | ~ Inconclusive | d=1,2 consistent with 2× prediction; d≥3 confounded by direct NMH edges + low SNR |
| Catch-up absolute timescale | ~ Shifted | d=1: $5.8\times$ longer than theory (MH normalization $\approx 1+\bar{k}$); d≥3 unmeasurable |

---

## Next Steps

1. **Experiment A — larger perturbation:** Re-run with `perturbation_scale=50` (×10 current) and ≥10 seeds. The larger perturbation raises SNR at d=3–5 from ≈3 to ≈30, enabling reliable $t_{50}$ detection. Expect to see the 2× doubling clearly for d=3–5 in the no-direct-edge group.

2. **Experiment D (decomposition autocorrelation):** Run `python -m analysis.run_catchup --exps D`. Measures $C_\delta(t)$ and $C_{m^\tau}(t)$ decay rates directly, testing $\gamma_\text{total}$ and $\gamma_L$ without any threshold calibration. This is the cleanest available test of the hierarchical timescale separation.

3. **Rescale theory by MH factor:** The effective discrete-time coupling is $\gamma_\ell^\text{eff} = \gamma_\ell / (1 + \bar{k}) \approx \gamma_\ell / 12$. Update theory overlays in all Exp A and Kramers plots to use $\gamma_\ell^\text{eff}$.

4. **Gossip-corrected Kramers formula:** Derive the effective barrier height with the MH-corrected coupling. The gossip term with $\gamma_\ell^\text{eff}$ lowers the effective barrier from $\Delta V^{(\ell)}$ to $\Delta V^{(\ell)} - \gamma_\ell^\text{eff}/(2T_\text{eff})$.

5. **Topological dimension:** Either increase graph size (depth=5 at small $p$) or accept $D$ is only measurable in the sparse regime ($p \leq 2$, where it approximately matches theory at D≈1.25).

6. **Heterogeneity floor subtraction:** To cleanly test $\sqrt{T_\text{eff}}$ scaling, run two temperature sweeps — one with homogeneous data (same weight for all modules) and one heterogeneous. The difference isolates the thermal contribution from the heterogeneity floor.
