# Test Summary: Regime C / Timesep Infrastructure
**Date:** 2026-05-23  
**Branch:** CanonicalTest  
**Python:** 3.9.6 · pytest 8.4.2

---

## Overall result

| Suite | Tests | Result | Time |
|---|---|---|---|
| `test_forest_fire.py` | 16 | **PASS** | — |
| `test_nmh_observables.py` | 17 | **PASS** | — |
| `test_ncp_sanity.py` | 14 | **PASS** | — |
| `test_natural_cascade_sanity.py` | 13 | **PASS** | — |
| **Total (new suites)** | **60** | **60 passed, 0 failed** | 370 s |

The full run command and output:

```
cd dssgd
pytest tests/test_forest_fire.py tests/test_nmh_observables.py \
       tests/test_ncp_sanity.py tests/test_natural_cascade_sanity.py \
       -v --tb=short
→ 60 passed in 370.56s
```

---

## What each suite validates

### `test_forest_fire.py` (16 tests) — NCP topology infrastructure

Tests the `ForestFireTopology` class that generates the NCP graphs used in experiments NCP-1 through NCP-5.

| Test | Validates |
|---|---|
| `test_step_returns_correct_types` | `step()` returns `(nx.Graph, np.ndarray)` of shape `(n, n)` |
| `test_graph_has_n_nodes` | Correct node count for n ∈ {10, 50, 100} |
| `test_connected_for_typical_params` | Connected graph for p_f ∈ {0.30, 0.35, 0.37} |
| `test_static_topology_same_every_round` | Static topology: same graph object across rounds |
| `test_mh_matrix_row_stochastic` | Row sums = 1 (doubly stochastic) |
| `test_mh_matrix_non_negative` | No negative mixing weights |
| `test_mh_matrix_symmetric` | W = Wᵀ (undirected graph) |
| `test_shell_assignment_returns_dict_with_all_nodes` | All n nodes covered by k-core decomposition |
| `test_shell_assignment_non_trivial_for_ncp_params` | max shell > 1 for p_f = 0.37 (non-trivial layered structure) |
| `test_shell_values_are_positive_integers` | Shell indices are positive integers (k-core ≥ 1) |
| `test_shell_assignment_agrees_with_networkx` | Matches `nx.core_number()` directly |
| `test_same_seed_same_graph` | Deterministic: same seed → identical edges and W |
| `test_different_seeds_different_graphs` | Different seeds produce different graphs |
| `test_invalid_p_f_zero_raises` | ValueError for p_f = 0 |
| `test_invalid_p_f_one_raises` | ValueError for p_f = 1 |
| `test_ensure_connected_produces_connected_graph` | Connectivity retry works across seeds |

---

### `test_nmh_observables.py` (17 tests) — Post-hoc observable functions

Tests the six observable functions in `nmh_observables.py` using **synthetic centroid trajectories** (no simulation required). All functions are pure — they operate on recorded `centroid_traj` arrays.

| Test | Observable | Validates |
|---|---|---|
| `test_cascade_depth_zero_when_no_flip` | `cascade_depth` | Returns 0 when no leaf enters B |
| `test_cascade_depth_one_when_level1_flips` | `cascade_depth` | Correct depth for level-1 flip |
| `test_cascade_depth_increases_with_more_flips` | `cascade_depth` | Depth 3 for a leaf at hierarchical distance 3 |
| `test_fraction_nan_when_no_leaves_at_level` | `fraction_reaching_level` | NaN for a level with no leaves |
| `test_fraction_one_when_all_at_level_flip` | `fraction_reaching_level` | 1.0 when all level-1 cousins flip |
| `test_fraction_zero_when_none_flip` | `fraction_reaching_level` | 0.0 when no level-1 cousin flips |
| `test_fraction_half_when_half_flip` | `fraction_reaching_level` | 0.5 when exactly half of level-2 cousins flip |
| `test_cascade_size_zero_at_t0` | `cascade_size` | 0 leaves in B at t=0 before any flip |
| `test_cascade_size_counts_b_leaves` | `cascade_size` | Correct count of B-leaves at a given round |
| `test_cascade_size_clamped_to_trajectory_length` | `cascade_size` | t_horizon beyond T-1 is clamped |
| `test_nucleation_prob_in_range` | `nucleation_prob_per_level` | Probabilities ∈ [0, 1] or NaN |
| `test_nucleation_prob_nonzero_after_transition` | `nucleation_prob_per_level` | Non-zero rate detected after a transition |
| `test_variance_decomp_returns_correct_levels` | `hierarchical_variance_decomposition` | Keys = {1, …, depth} |
| `test_variance_decomp_values_are_finite` | `hierarchical_variance_decomposition` | All values finite |
| `test_variance_decomp_t_range` | `hierarchical_variance_decomposition` | t_range selects a time window |
| `test_detailed_balance_ratio_nan_when_no_backward_transitions` | `detailed_balance_ratio` | NaN/inf when no B→A transitions (expected for b > 0) |
| `test_detailed_balance_ratio_approx_one_for_symmetric` | `detailed_balance_ratio` | Ratio in (0.1, 10) for equal-rate synthetic flip process |

---

### `test_natural_cascade_sanity.py` (13 tests) — Regime C simulation runner

Tests `run_natural_cascade_simulation()`, the core runner for NMH experiments NMH-1 through NMH-7. Verifies the **no-force-flip** natural cascade mechanism and post-hoc nucleation detection.

| Test | Validates |
|---|---|
| `test_smoke_completes` | Runner completes for depth=2, leaf_size=2, n_meas=30 |
| `test_centroid_traj_shape` | `centroid_traj` shape = `(n_meas+1, n_leaf_types, d_param)` |
| `test_warmup_ok_standard_params` | Warmup converges to basin A for standard Regime C params |
| `test_symmetric_warmup_nmh7_does_not_crash` | b=0 (NMH-7) runs without error; warmup_ok type is bool |
| `test_detects_nucleation_given_long_enough_run` | Nucleation detection runs without error; output types correct |
| `test_flip_table_structure` | All required keys present in every flip_table entry |
| `test_flip_table_excludes_nucleation_leaf` | Nucleation leaf absent from flip_table |
| `test_t_flip_relative_consistent` | `t_flip_relative = t_flip_absolute − t_nucleation` |
| `test_per_leaf_loss_nmh6_runs` | Per-leaf (a_i, b_i) loss params (NMH-6) run without error |
| `test_per_leaf_loss_wrong_length_raises` | ValueError for mismatched per_leaf_loss_params length |
| `test_picklable` | `NaturalCascadeRun` pickles and restores with identical arrays |
| `test_regime_field_is_valid` | `run.regime` ∈ {'I', 'II', 'III'} |
| `test_ell_c_field_is_int` | `run.ell_c` is a non-negative integer |

---

### `test_ncp_sanity.py` (14 tests) — NCP simulation runner and observables

Tests `run_ncp_simulation()` and the three shell-level observable functions. Uses small graphs (n ≤ 25).

| Test | Validates |
|---|---|
| `test_smoke_completes` | NCP runner completes for n=20 nodes |
| `test_centroid_traj_shape` | Shape `(n_meas+1, n_nodes, d_param)` |
| `test_shell_traj_keys_match_shell_assigns` | `shell_traj` keys = `set(shell_assigns.values())` |
| `test_shell_traj_shapes` | Each shell trajectory has shape `(n_meas+1, d_param)` |
| `test_graph_stats_populated` | `graph_stats` has `n_nodes`, `n_shells`, `mean_degree`, `max_shell` |
| `test_warmup_ok_type` | `warmup_ok` is a bool |
| `test_clamped_shell_nodes_near_B` | Clamped-shell nodes (NCP-2 mechanism) have params > 0.7 at final round |
| `test_flip_table_structure` | Required keys in every entry |
| `test_flip_table_excludes_nucleation_node` | Nucleation node absent from flip_table |
| `test_picklable` | `NCPRun` pickles and restores |
| `test_shell_modal_basin_returns_valid_labels` | Labels ∈ {'A', 'B', 'X'} |
| `test_decoupling_chi_in_range` | χ ∈ [0, 1] for valid shell pair |
| `test_decoupling_chi_nan_for_missing_shell` | NaN for absent shell index |
| `test_cmi_non_negative` | Conditional mutual information ≥ 0 |

---

## Bug found and fixed

**`verify_bistable_loss` fails for b=0 (NMH-7 symmetric basins).**

With b=0 both basin centres A and B are exact critical points of the quartic potential, so L(A) = L(B) = 0. The existing check `L(B) >= L(A)` raised `ValueError` for this valid symmetric configuration.

**Fix:** In `natural_cascade.py`, skip `verify_bistable_loss` entirely when `b = 0.0`. The symmetric quartic with a > 0 is a valid bistable landscape by construction; the inequality check is meaningless when both wells have equal depth.

---

## Scope note

The **existing** test suites (`test_active_escape_sanity.py`, `test_catchup_sanity.py`, `test_consensus_convergence.py`) were not re-run in this session. No existing files were modified, so these suites should be unaffected. The most recent prior run of those suites (2026-05-22) recorded 30/30 passes for `test_active_escape_sanity.py`.

---

## New infrastructure summary

| File | Role |
|---|---|
| `dssgd/topology/forest_fire.py` | `ForestFireTopology` (NCP graphs, k-core shell decomposition) |
| `analysis/natural_cascade.py` | Regime C runner: no force-flip, post-hoc nucleation, per-leaf loss |
| `analysis/natural_cascade_experiments.py` | NMH-1/1b/2/3/4/5/6/7 experiment factories |
| `analysis/nmh_observables.py` | 6 post-hoc observables for NMH experiments |
| `analysis/ncp_runner.py` | NCP simulation runner: per-node agents, shell clamping |
| `analysis/ncp_observables.py` | `shell_modal_basin`, `decoupling_chi`, `conditional_mutual_information`, `propagation_matrix` |
| `analysis/ncp_experiments.py` | NCP-1/2/3/4/5 experiment factories |
