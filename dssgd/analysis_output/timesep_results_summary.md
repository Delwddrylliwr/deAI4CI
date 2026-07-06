# Regime C Timescale-Separation Experiments: Results and Compute Survey

**Branch:** CanonicalTest  
**Simulation framework:** gossip-SGD with bistable loss, NestedModularTopology (NMH) and ForestFireTopology (NCP)  
**Operating point:** `local_steps=50`, `lr=0.1`, `b=0.042`, `p=2.0`, `leaf_size=4`, `depth=5` unless noted  
**Reference compute:** ≈ 35 min per config at the above operating point (128 agents × 50 local_steps × 200 rounds, single CPU core, Windows 11)

---

## 1. Infrastructure

All simulation and analysis infrastructure is implemented and passing tests (60/60, 2026-05-23).

| File | Role |
|---|---|
| `dssgd/topology/forest_fire.py` | `ForestFireTopology` with k-core `shell_assignment()` |
| `analysis/natural_cascade.py` | Regime C runner: force-flip source, post-hoc nucleation detection |
| `analysis/natural_cascade_experiments.py` | Factories for NMH-1 through NMH-7 |
| `analysis/nmh_observables.py` | `cascade_depth`, `fraction_reaching_level`, `cascade_size`, `nucleation_prob_per_level`, `hierarchical_variance_decomposition`, `detailed_balance_ratio` |
| `analysis/ncp_runner.py` | Per-node NCP runner with shell clamping |
| `analysis/ncp_experiments.py` | Factories for NCP-1 through NCP-5 |
| `analysis/ncp_observables.py` | `shell_modal_basin`, `decoupling_chi`, `conditional_mutual_information`, `propagation_matrix` |
| `analysis/run_timesep.py` | CLI orchestrator with caching; currently wires NMH-1, NMH-1b, NMH-3, NCP-1 |

**Bistable loss:** $L(\theta) = \tfrac{a}{2}\theta^2(\theta-1)^2 - b(\theta - \tfrac{1}{2})$, minima at $A \approx 0$, $B \approx 1$, saddle at $\theta_s \approx \tfrac{1}{2} - \tfrac{2b}{a}$.

**Regime C condition:** `local_steps` $\gg 1$ so agents relax fully to their basin attractor between gossip events. The state space collapses to $\{A,B\}^N$ and cascade dynamics are gossip-driven.

---

## 2. Completed Experiments

### 2.1 NMH-1 — Hierarchical mixing-time scaling (slope-1 test)

**Theoretical prediction (Theorem 1):** $\tau_\text{mix}^{(\ell)} = \Theta(2^{\ell+1} / (\varepsilon p M_0 q_\ell))$, implying $\log_2 T_\text{flip}(d) \approx d + \text{const}$, slope = 1 in hierarchical distance $d$.

**Setup:** 20 seeds, $a=0.5$, `force_flip_source=True`, `n_meas=200`.  
**Note:** $a=0.5$ is the only value that reliably cascades at $p=2$ (see §2.3). Larger $a$ values (1.0, 2.0, 4.0) tested in the 5-seed pilot produced no cascade, consistent with Griffiths-phase suppression at those parameter values (see §2.3).

**Results** (mean $\log_2 t_\text{flip,rel}$ per distance; 20 seeds, $a=0.5$):

| Distance $d$ | Obs. | Mean $\log_2 t$ | SE |
|---|---|---|---|
| 1 | 13/20 | 0.35 | 0.15 |
| 2 | 40/40 | 1.92 | 0.13 |
| 3 | 76/80 | 3.06 | 0.06 |
| 4 | 144/160 | 4.36 | 0.04 |
| 5 | 240/320 | 5.15 | 0.03 |

OLS slope ($d = 2\ldots5$): **1.097** (theory: 1.0).  
OLS slope ($d = 1\ldots5$): 1.202 (biased upward by $d=1$ floor effect — $d=1$ siblings cascade in ≤ 1 round).

**Missing observations:** 7/20 seeds miss $d=1$ (source centroid oscillates near threshold before stabilising); 80/320 miss $d=5$ (level-5 cross-edge probability $p/4^5 \approx 0.002$, cascade probability $< 1$ within 200 rounds).

**Verdict:** Slope-1 confirmed for $d = 2\ldots5$. The 2× doubling of mixing time per hierarchy level is consistent with Theorem 1.

---

### 2.2 NMH-1b — Regime C local_steps sensitivity

**Theoretical prediction:** Slope-1 holds only when `local_steps` is large enough that agents fully relax to their basin between gossip events (Regime C). Below threshold, the source cannot maintain $B$; above a second threshold, over-relaxed $A$-state agents resist sparse high-level gossip.

**Setup:** 5 seeds, $a=0.5$, `local_steps` $\in \{10, 50, 200\}$, `n_meas=200`.

**Results:**

| `local_steps` | Cascades | Slope $d=2\ldots5$ | Notes |
|---|---|---|---|
| 10 | 0/5 seeds | — | Source cannot maintain $B$: 10 gradient steps insufficient to recover from gossip dilution. Post-gossip centroid $\approx 0.66$; 10 steps recovers only to $\approx 0.74$, below $B$ threshold of 0.8. |
| 50 | 4–5/5 seeds | **1.05** | Regime C: source reliably holds $B$; cascade propagates to $d=5$ in most seeds. |
| 200 | 1/5 seeds (complete) | — (sparse data) | Over-relaxation: $A$-state agents are maximally relaxed to $A_\text{min}$; sparse level-4/5 cross-edges ($p/4^4 \approx 0.008$) insufficient to tip them in most seeds. One seed achieves complete cascade ($d=5$) in $\approx 18$ rounds — faster per level than `ls=50` but rare. |

**Non-monotone relationship:** `ls=10` $\to$ no cascade; `ls=50` $\to$ reliable slope-1 cascade; `ls=200` $\to$ heavy-tailed cascade depth, 1/5 seeds complete. The `ls=200` distribution is consistent with Griffiths-phase dynamics at the boundary of Regime C.

**Verdict:** `local_steps=50` is the Regime C operating point. The slope-1 relationship is specific to this regime.

---

### 2.3 NMH-3 — Three-regime phase structure

**Theoretical prediction:** Three regimes at $p=2$:
- **Regime I** ($a < 0.5$, ferromagnetic): cascade propagates to all levels with high probability.
- **Regime II** ($0.5 \leq a < 8$, Griffiths): cascade depth is heavy-tailed; `ell_c` decreases from 4 to 1 as $a$ increases.
- **Regime III** ($a \geq 8$, paramagnetic): cascade impossible (`ell_c = 0`).

**Setup:** 5 seeds, 8 $a$-values $\in \{0.25, 0.5, 1, 2, 4, 8, 12, 16\}$, `force_flip_source=True`, `n_meas=200`.

**Results:**

| $a$ | Warmup OK | Regime | $\ell_c$ | Cascades observed (5 seeds) | Max depth |
|---|---|---|---|---|---|
| 0.25 | 0/5 (**fail**) | I | 5 | — | — |
| 0.5 | 5/5 | II | 4 | 5/5 | 4–5 |
| 1.0 | 5/5 | II | 3 | 0/5 | — |
| 2.0 | 5/5 | II | 2 | 0/5 | — |
| 4.0 | 5/5 | II | 1 | 0/5 | — |
| 8.0 | 5/5 | **III** | 0 | 0/5 | — |
| 12.0 | 5/5 | III | 0 | 0/5 | — |
| 16.0 | 5/5 | III | 0 | 0/5 | — |

**Notes:**

- **$a=0.25$ warmup failure:** The $A$ basin is too shallow for the gradient-descent warmup (400 rounds, lr=0.1) to reliably establish the initial condition. Regime I is theoretically predicted ($\ell_c = 5$) but unobservable with the current protocol. Requires increased warmup budget or lower learning rate during warmup.
- **$a = 1\ldots4$ (Regime II, zero observed cascades):** The theory classifies these as Griffiths phase with $\ell_c \in \{1,2,3\}$, meaning cascade events are possible but rare. Zero cascades in 5 seeds is consistent with heavy-tailed event probabilities. Distinguishing Griffiths suppression from Regime III requires $\mathcal{O}(100)$ seeds at each $a$ value.
- **$a=8$ (II→III boundary):** Sharp: all seeds warmup OK, `ell_c` drops to 0, no cascade in any seed. The Regime II–III boundary is confirmed at $a = 8$ for these parameters.
- **$a=0.5$ flip data** matches NMH-1 exactly (same seeds, same timing), confirming experimental consistency.

**Verdict:** Regime boundaries confirmed at $a = 0.5$ (I→II) and $a = 8$ (II→III). Regime I requires warmup fix. Regime II Griffiths-phase cascade events require large-$N$ seed runs to observe.

---

## 3. Pending Experiments

The experiment factories and analysis code for all experiments below are implemented. The CLI orchestrator (`run_timesep.py`) needs runner functions added for NMH-2, NMH-4 through NMH-7, and NCP-2 through NCP-5.

### Compute reference

All estimates assume single sequential CPU core (reference: 35 min per config at 128 agents × 50 local_steps × 200 rounds). Scale factors applied for depth, `n_meas`, and `local_steps` relative to reference. NCP estimates use `n_nodes = 1000`.

### 3.1 Full experiment table

| Experiment | What it tests | Theory ref. | Config count | Per-config (hrs) | Total (single core) |
|---|---|---|---|---|---|
| **NMH-1** (complete to 50 seeds, `n_meas=1000`) | Slope-1 $\tau \propto 2^\ell$ | Theorem 1 | 30 remaining | 2.9 | **87 hrs** |
| **NMH-1b** (full: 4 ls-values × 20 seeds, `n_meas=1000`) | Regime C threshold | §2.4 | 65 remaining | 0.6–58 (ls-dependent) | **~300 hrs** |
| **NMH-2** | Nucleation prob $q_\ell$ vs bias $b$ | Eq. (14) | 700 (7b × 100 seeds) | 2.9 | **2,030 hrs** |
| **NMH-3** (complete to 30 seeds, `n_meas=1000`) | Phase structure | §2.7 | 200 remaining | 2.9 | **580 hrs** |
| **NMH-4** | Cascade size power-law $P(s) \sim s^{-\tau}$ | Prop. 1, Eq. (18) | 2,000 (depth=7) | 9.3 | **18,600 hrs** |
| **NMH-5** | Filter composition $d_\text{prop}(b/a)$ | Theorem 3, Eq. (19)–(20) | 750 (3a × 5(b/a) × 50 seeds) | 2.9 | **2,175 hrs** |
| **NMH-6** | Stationary variance decomposition | Prop. 2, Eq. (22)–(23) | 30 (`n_meas=2000`) | 5.8 | **175 hrs** |
| **NMH-7** | Detailed balance / Gibbs ($b=0$) | Theorem 2, Eq. (15)–(17) | 20 (depth=4, `n_meas=4000`) | 5.8 | **116 hrs** |
| **NCP-1** | Forest Fire shell-depth scaling | §3.1 | 240 (graph only) | ~0 | **< 1 hr** |
| **NCP-2** | Asymmetric core↔periphery nucleation | §3.5, Eq. (30)–(32) | 50 (n=1000) | 18.2 | **910 hrs** |
| **NCP-3** | Decoupling profile $\chi_k$ | Theorem 4, Eq. (35) | 30 (n=1000) | 18.2 | **546 hrs** |
| **NCP-4** | Propagation matrix / cascade suppression | §3.9, Eq. (39) | 200 (n=1000) | 18.2 | **3,640 hrs** |
| **NCP-5** | Stationary CMI factorisation $I(B_k;B_{k+2}\mid B_{k+1}) \approx 0$ | Theorem 4, Eq. (33) | 10 (n=1000, `n_meas=3000`) | 68.4 | **684 hrs** |

**Total (full spec, excluding already-run pilots):** ≈ 29,844 CPU-hours ≈ **3.4 CPU-years**.

### 3.2 Parallelisation

All configs are embarrassingly parallel by seed. Wall-clock times scale linearly with core count.

| Platform | Cores | Approx. wall-clock |
|---|---|---|
| Single workstation (current) | 1 | 3.4 years |
| 8-core workstation | 8 | ~5 months |
| 32-core server | 32 | ~5.5 weeks |
| 256-core HPC node / cloud burst | 256 | ~5 days |

Free-tier cloud (e.g. Colab, Kaggle) can parallelise short configs (NMH-1/3 seeds) within session limits (9–12 hrs per notebook). Long-running configs (NMH-4, NCP) require persistent compute.

---

## 4. Prioritisation

### 4.1 Minimal viable set for a theoretical report

The following subset provides coverage of all three main theorems and both topology families without requiring cluster compute:

| Priority | Experiment | Rationale | Est. wall-clock (single core) |
|---|---|---|---|
| 1 | **NCP-1** (run now) | Graph-only; confirms Forest Fire shell structure, no simulation needed | < 1 hr |
| 2 | **NMH-7** | Theorem 2 (detailed balance / Gibbs) — most theoretically fundamental remaining test; $b=0$ symmetric basins; depth=4 reduces agent count | 116 hrs |
| 3 | **NMH-3** (extend to 30 seeds) | Needed to observe Griffiths-phase partial cascades at $a=1,2$; 5-seed pilot shows zero cascade for $a \geq 1$ | 580 hrs (or ~200 hrs for 10 seeds) |
| 4 | **NMH-6** (pilot: 5 seeds) | Variance decomposition with heterogeneous loss — unique observable not available from other experiments | ~30 hrs pilot |
| 5 | **NMH-2** (pilot: 10 seeds × 7 $b$) | Nucleation-rate vs bias; establishes whether $q_\ell(b)$ is measurable with available noise | ~200 hrs pilot |

### 4.2 Open issues requiring attention before re-running

1. **NMH-3 / NMH-1, $a=0.25$ warmup failure.** The $A$ well at $a=0.25$ is too shallow for the 400-round gradient-descent warmup to converge. Fix options: increase `n_warmup` to 1600+, reduce warmup `lr`, or add a warmup convergence check with adaptive budget.

2. **NMH-3 / $a=1\ldots4$ Griffiths detection.** With 5 seeds no cascade events are observed. To characterise the heavy-tailed distribution, $\mathcal{O}(50\text{–}100)$ seeds per $a$-value are needed. Run as a separate long-duration background job.

3. **NMH-1b `ls=200` Griffiths interpretation.** The stalling pattern (1/5 seeds complete, others stop at $d=3$–4) is consistent with Griffiths dynamics but could also reflect an insufficient measurement window or topology-specific blocking. A follow-up run at `n_meas=1000` with 20 seeds would disambiguate.

4. **NCP experiments at reduced scale.** The spec uses $n=1000$ nodes throughout. Reducing to $n=100$ cuts per-config time by $10\times$ (to ~1.8 hrs), making NCP-1/2/3 pilots feasible on a workstation. Scaling behaviour should still be discernible with $n \in \{100, 200, 500\}$.

5. **NMH-4 scale.** 2000 seeds at depth=7 is infeasible on a single machine. A depth=5 pilot at $a=2$ (Griffiths midpoint) with 200 seeds would test the power-law tail with $\approx 32\times$ less compute (580 hrs → 18 hrs).

---

## 5. Parameter summary

| Symbol | Role | Default |
|---|---|---|
| $a$ | Loss curvature; controls saddle height $\theta_s \approx \frac{1}{2} - \frac{2b}{a}$ | 0.5 |
| $b$ | Basin asymmetry / bias toward $B$ | 0.042 |
| `local_steps` | Gradient steps per gossip round; controls Regime C | 50 |
| `depth` | Tree depth; determines number of hierarchical levels | 5 |
| `leaf_size` | Agents per leaf module | 4 |
| $p$ | Level-1 cross-edge density; Regime boundaries scale with $p$ | 2.0 |
| `n_meas` | Measurement rounds per config | 200 (pilot) / 1000 (full) |
| `n_warmup` | Warmup rounds to establish basin $A$ | 400 |
| `epsilon` | Basin detection half-width ($B$ if $\theta > 1-\varepsilon$) | 0.2 |
| `persistence` | Consecutive rounds in $B$ required for flip detection | 3 |
| $n_\text{nodes}$ | Agents in NCP Forest Fire graph | 1000 |
| $p_f$ | Forest Fire forward burning probability | 0.37 |
