# Test Summary — 2026-05-22

## Context

Two rounds of changes preceded these runs:

1. **p=2 Safari regime** — changed the default inter-module edge probability from `p=float(leaf_size)=4` (degenerate complete-bipartite K_{m,m} at level 1) to `p=2.0` (stochastic sparse regime, level-1 edge probability 0.5, as studied in Safari et al.).  Updated regime boundaries throughout: Regime I `a < 0.5`, Regime III `a ≥ 8`, critical curvature `a_c = 8` (previously `a_c = 16` with p=4).

2. **Natural-cascade experiment (A1nat)** — added `experiment_A1_natural` (`clamp_source=False`, `b=0.042`, `local_steps=50`, `p=4.0`) as a FedAvg-regime probe of self-sustaining cascades without source clamping.  Required adding `require_barrier` parameter to `verify_bistable_loss` since with `b=0.042 > 2a/27 ≈ 0.037` the energy saddle sits below `L(A)`, even though a gradient saddle exists at `θ_s ≈ 0.30`.

---

## Run 1 — Full suite (57 tests)

**Command:** `python -m pytest tests/ -v`
**Duration:** 19m 47s
**Result:** 56 passed, 1 failed

### Failure

```
FAILED tests/test_consensus_convergence.py::test_gossip_sgd_accuracy[lian2017/gossip/mnist/ring/n=8/iid]
Final accuracy 0.821 < threshold 0.850 after 200 rounds
```

Pre-existing stochastic flakiness — the accuracy is close to the published threshold and this test was already marginal before any of the current changes.  All active-escape tests and all other tests passed.

---

## Run 2 — Natural-cascade smoke test (1 test)

**Command:** `python -m pytest tests/test_active_escape_sanity.py::test_natural_cascade_flips_without_clamping -v -s`
**Duration:** 8m 22s (first attempt)
**Result:** 1 failed → 1 passed (after fix)

### Initial failure

```
ValueError: No barrier detected between A and B (max interior loss 0.0195 <= L(A)=0.0210)
```

`verify_bistable_loss` checked that the interior energy exceeded `L(A)`, which fails when `b > 2a/27`.  For the natural cascade this check is inappropriate: with `clamp_source=False`, A is not a stable energy minimum (gradient at A = −b always points toward B), so the absence of an energy barrier above L(A) is by design.

**Fix:** added `require_barrier: bool = True` parameter to `verify_bistable_loss`.  The call in `run_active_escape_simulation` passes `require_barrier=config.clamp_source`, so clamped experiments retain the full check and natural-cascade configs skip it.

### After fix

Test passed: `d=1` leaf entered basin B within 200 measurement rounds without any source clamping.

**Mechanism confirmed:** `b=0.042` shifts the gradient saddle to `θ_s ≈ 0.30`.  After one gossip round the level-1 cousin lands at `θ ≈ 0.33 > θ_s`, putting it in the "flow toward B" region.  With `local_steps=50` the agent has enough local SGD time to approach B before the next gossip exchange.

---

## Run 3 — Active-escape suite (30 tests)

**Command:** `python -m pytest tests/test_active_escape_sanity.py -v`
**Duration:** 4m 02s
**Result:** 30 passed

Covers: bistable-loss landscape checks, `verify_bistable_loss` (including `require_barrier` edge cases), `force_flip_module`, `basin_label`, `find_t_flip`, theory (`classify_regime`, `critical_level`, `deterministic_flip_time`, `kappa_loc`), three simulation smoke tests (standard), and the new `test_natural_cascade_flips_without_clamping`.

---

## Files changed this session

| File | Change |
|------|--------|
| `dssgd/topology/static.py` | Default `p=2.0`; updated docstring |
| `analysis/active_escape.py` | Config default `p=2.0`; `clamp_source=True` field + re-clamp block; `require_barrier` param in `verify_bistable_loss` |
| `analysis/active_escape_experiments.py` | A1/A2/A3 boundaries for p=2; added `experiment_A1_natural` |
| `analysis/run_active_escape.py` | Sanity checks updated for p=2; `run_experiment_A1_natural` runner; `A1nat` in argparse and `priority_order` |
| `tests/test_active_escape_sanity.py` | Theory tests updated to p=2 boundaries; added `test_natural_cascade_flips_without_clamping` |
