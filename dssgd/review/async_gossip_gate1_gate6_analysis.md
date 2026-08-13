# Async-gossip cross-gate analysis: Gate 1 vs Gate 6

*Comparing the async-gossip failures in Gate 1 (`gate1_review.json` + CSVs) against the Gate 6 results (`gate6_review.json` + CSVs), which so far has only been run under async gossip. Source data: `gate1_*`, `gate6_*` in this directory. Theory reference: `paper1_PDMP_wDAG_wData.md`.*

## Bottom line

- **Mixing exponent (Prop. 4.2 / Remark 4.3):** Gate 1's unexplained slope failure is now explained by Gate 6, and the explanation is favorable to the theory.
- **Containment / level-matching (Thm. 4.5, Prop. 4.9):** Gate 1 and Gate 6 each show an invariant, parameter-insensitive result — but in *opposite* directions (always-propagates vs. never-propagates). This does not corroborate or refute the theory; it points at harness/clamping configuration issues that need to be resolved before either result is informative.
- **NCP shell scaling (structural, not dynamical):** unaffected by the sync/async question, and remains the most solid confirmed result across both gates.

---

## 1. Mixing exponent: Gate 1's failure is explained by Gate 6

**Gate 1 (`gate1_nmh1_slopes.csv`):** slope ≈ 0.51–0.59 across every curvature `a` (0.5, 1.0, 2.0, 4.0) — flat, `a`-independent, and far from the target of ~1. `slope_confirmed: false`; gate1's own notes guessed at the cause ("check local_steps or Regime C conditions").

**Gate 6 (`gate6_e11_slopes.csv`):** directly varies scheduling protocol at a matched operating point:

| protocol | staleness_bound | slope |
|---|---|---|
| synchronous | 0 | **1.061** |
| async_poisson | 0 | **0.540** |
| bounded_staleness | 2 | 0.465 |
| bounded_staleness | 5 | 0.440 |
| bounded_staleness | 10 | 0.419 |

`async_poisson`'s slope (0.540) lands almost exactly on Gate 1's whole range, while `synchronous` (1.061) matches the previously-validated Annex A.1 figure (1.097). This is precisely Remark 4.3's predicted mechanism: free per-edge asynchrony causes multi-seed immigration, which flattens the geometric (slope-1) mixing law toward the linear (multi-seed) regime.

**Reframing:** Gate 1 was not falsifying Proposition 4.2 — it was (apparently unknowingly) run under the free-asynchrony regime the whole time, for which slope ≈0.5 *is* the predicted outcome, not a failure. The `a`-independence in Gate 1 is consistent with this too: scheduling, not curvature, is the dominant lever on the exponent. Proposition 4.2 carries H-sched (round-synchronous scheduling) as a standing assumption; Gate 1 didn't satisfy it.

**Caveat — the repair path is not yet demonstrated.** Gate 6's bounded-staleness rows do *not* trend back toward slope 1 as the bound tightens (2 → 0.465, 5 → 0.440, 10 → 0.419) — all three sit *below* pure async (0.540). Remark 4.3 proposes bounded staleness / seed-locking as a way to restore slope-1 stratification under asynchrony; that repair has not been shown empirically yet, and the non-monotonic ordering relative to pure async is itself unexplained (possibly a thin-slope-fit artifact given `n_distances=4`).

---

## 2. Containment / level-matching: two invariant results in opposite directions

**Gate 1 (`gate1_nmh3_depths.csv`):** `mean_depth = 5.0`, `std_depth = 0.0` at **every** `a` from 0.5 to 16.0 (25 runs each) — full-depth (ceiling) propagation, zero variance, regardless of curvature. This directly contradicts the paper's own hand-validated Annex A.3 result at the same parameter grid, which reports a sharp collapse to zero cascades at `a ∈ {8, 12, 16}`. Gate 1 is reproducing the *opposite* of a result the paper claims to have already established.

**Gate 6 (`gate6_e12a_containment.csv`):** `mean_d_max = 0.0`, `frac_d_max_eq_G = 0.0` for every `G ∈ {1, 2, 3}` (50 runs each) — zero (floor) propagation, regardless of generality level. This is the direct test of Theorem 4.5's attainment half (a favored innovation should reach depth `G(b)` with probability → 1), and it fails outright: not "wrong depth," but no propagation at all under any tested condition.

**Why these don't corroborate each other via "it's just async":** if scheduling alone explained both, you'd expect *some* response to the swept parameter in each case — attenuated stratification, not a hard invariant 0 or a hard invariant ceiling. Instead:

- Gate 1 (globally-favored single innovation) *always* fully propagates, even where the energy barrier should suppress it entirely.
- Gate 6 E12a (level-heterogeneous innovation gated by generality `G`) *never* propagates, even within its own favored scope.

This pattern is more consistent with two separate, opposite-direction configuration issues than with a single scheduling effect:

- **Gate 1 / nmh3:** plausibly over-clamping — if the source is force-reseeded every round (`clamp_source=True` in the standard active-escape config), a persistent source under async scheduling can eventually punch through any finite barrier by brute-force seed accumulation (cf. Prop. D.3.2 / Remark 11.4: asynchrony erodes the ratchet exponent from `m−1` toward `m−σ_sched`). That would produce "always reaches max depth" regardless of `a`. Consistent with gate1's own hypothesis of Regime-C contamination.
- **Gate 6 / E12a:** plausibly under-clamping / insufficient tilt at the boundary — consistent with the noted ~1600× gossip-vs-gradient imbalance for unclamped runs, under which a seed reverts before it can fix.

**Action before trusting either result:** check whether `nmh3` (gate1) and `E12a` (gate6) use matched `clamp_source` / relaxation-horizon settings. As it stands, neither result should be read as evidence for or against Theorem 4.5 / Proposition 4.9 — both look like they're testing harness configuration rather than the theory.

---

## 3. Unaffected by the sync/async question: NCP shell scaling

`gate1_ncp1_scaling.csv` (mean shell count vs. `N` and `p_f`) is a static topology-generation result, not a dynamics result, so it isn't subject to the scheduling confound above. It remains consistent with Annex A.4: `k_max` grows with `N` and increases with `p_f`. This continues to be the most solid confirmed result across both gates.

---

## 4. Recommended next steps

1. Re-run Gate 1's `nmh1` slope sweep under explicit `synchronous` scheduling to confirm it converges to the Annex A.1 figure (~1.1) — this would close out the mixing-exponent question cleanly.
2. Investigate the non-monotonic bounded-staleness ordering in Gate 6 E11 (all three bounds below pure async) — likely needs more distance points (`n_distances` > 4) before drawing conclusions about the repair mechanism.
3. Audit clamping/relaxation-horizon configuration in Gate 1 `nmh3` and Gate 6 `E12a` against each other and against Prop. 3.6's "operating wedge" — the ceiling-saturation and floor-saturation results are each other's mirror image and are unlikely to both be correct readings of the theory.
4. Re-run `E12a` (and Gate 1 `nmh3`) only after (3) is resolved; until then, treat Theorem 4.5 / Proposition 4.9 as untested by either gate, not as failed.
