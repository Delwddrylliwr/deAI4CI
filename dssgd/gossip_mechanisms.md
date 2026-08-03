# The three gossip mechanisms, and why pairwise sync got its own name

This repo distinguishes three gossip mechanisms, each with its own `gossip_protocol`
string and experiment-name/pkl-directory suffix (`dssgd.protocols.gossip.protocol_suffix`
/ `protocol_from_suffix`):

| `gossip_protocol` | Class | Suffix | What it does |
|---|---|---|---|
| `"async_poisson"` | `AsynchronousGossip` | *(none)* | Poisson-timed, one-sided pairwise pull events, sequential within a round. |
| `"sync_pairwise"` | `SynchronousPairwiseGossip` | `SP` | Round-synchronous: a maximal matching each round, each matched pair doing the *same* pairwise pull `AsynchronousGossip` uses, all reading from one pre-round snapshot. |
| `"sync_neighbourhood"` | `GossipAveraging` | `SN` | Round-synchronous: every agent simultaneously averages with its *whole* current neighbourhood + itself — a simultaneous m-way mean, not a pairwise kick. |

`bounded_staleness` (E11 only) is a fourth, special-purpose variant of `AsynchronousGossip`
with a per-edge staleness lock; it doesn't get a directory suffix of its own — E11 encodes
it in the run name (`name=f"E11/proto={protocol}/stale={staleness}/..."`) since it sweeps
multiple scheduling classes within one experiment.

## Why this needed sorting out

The paper's theory (`paper1_PDMP_wDAG_wData.md`, Remark 4.3) draws a specific scheduling
distinction: **round-synchronous scheduling of pairwise kicks** (the "H-sched" class) is
predicted to *preserve* the level-matching filter (Theorem 4.5) by construction, because
every boundary seed's fixation contest resolves before the next one arrives. That is a
claim about *pairwise* dynamics under synchronous timing — it says nothing about
simultaneous multi-neighbour averaging, which is a structurally different update rule
(Lemma 6.1's basin-*destroying* mechanism, the one E9's centralised-aggregator test
deliberately uses).

For a long stretch of this repo's history, both of these were called "synchronous" and
given the same "S" suffix:

1. Originally, `"synchronous"`/`"S"` meant `GossipAveraging` (the m-way mean) — a bug: it
   was standing in for the H-sched pairwise class the theory actually specifies, and
   produced NMH-2/E12a/E12b results that looked like containment/nucleation failing
   completely under "sync," when what had actually failed was the wrong mechanism being
   tested.
2. That bug was fixed by swapping the underlying class to `SynchronousPairwiseGossip` —
   but the string and suffix (`"synchronous"`/`"S"`) were left unchanged, so the *label*
   silently started meaning something different from what it meant in any data generated
   before the fix. A `.pkl` file named `NMH2S/...` could now mean either mechanism,
   distinguishable only by checking when it was generated — a real reproducibility
   hazard, and the trigger for this note.

## The fix

`"synchronous"`/`"S"` are retired outright (raise, not silently guessed) and every
sync-family experiment must now say explicitly which mechanism it means:
`"sync_pairwise"` (suffix `SP`) or `"sync_neighbourhood"` (suffix `SN`). This makes three
things true going forward:

- Any `SP`- or `SN`-suffixed file is unambiguous regardless of when it was generated —
  the label *is* the mechanism, not a proxy for "whatever sync meant at the time."
- `sync_neighbourhood` (`GossipAveraging`) is preserved as a legitimate, separately
  runnable mechanism in its own right — it's still what E9's basin-destruction test uses,
  and it's now available as an explicit third arm for any experiment that wants to
  compare "uniform averaging" against both async and pairwise-sync directly, rather than
  being an accidental historical stand-in.
- The phase-ID convention in `generate_queue.py` (`queue/phase2s`, etc.) is **also**
  renamed, `"Xs"` → `"Xsp"` — see the dedicated section below; it turned out this level
  had the exact same ambiguity problem as the experiment suffix, not just a cosmetic
  mismatch with it.

## Practical notes

- `dssgd.protocols.gossip.protocol_suffix(gossip_protocol)` and the inverse
  `protocol_from_suffix(suffix)` are the single source of truth for this mapping;
  every experiment factory and review script (`check_phase1/2/3/5/6.py`,
  `compare_async_sync.py`) uses them rather than repeating the three-way logic locally.
- `check_phase*.py`'s `--suffix` argument now only accepts `""`, `"SP"`, or `"SN"` — the
  bare `"S"` is rejected with an error pointing back to this note, rather than silently
  guessing which mechanism a legacy file means.
- Pre-existing `"S"`-suffixed data from before this fix is **not** automatically
  reinterpretable as either mechanism — you have to know, from when/how it was generated,
  which one it actually is, and treat it as `SP` or `SN` accordingly (or just re-run it
  under the new explicit naming).

## The phase-ID level had the same ambiguity — and it was worse

The experiment-suffix fix above was originally believed to be sufficient, on the
assumption that `generate_queue.py`'s phase-ID convention (`"1s"`, `"2s"`, ... `"6s"`)
was a stable, unambiguous higher-level label that had always meant `sync_pairwise`.
Checking the actual local pkl/review archives against git history disproved that:

- Every locally-pulled pkl snapshot for phases 1, 2, and 6's sync tracks shares the
  same **2026-07-24** pull batch (file mtimes within an 8-minute window per phase) —
  and `SynchronousPairwiseGossip` doesn't appear in git history until commit `7494a99`,
  committed **2026-07-25**. So all three phases' raw pkl data, read strictly by
  generation time, is genuinely `sync_neighbourhood` (the pre-fix `GossipAveraging`
  mechanism), regardless of what any directory was named at the time.
- Phase 2's *review* JSON told a different, partial story: `review/phase2s` (Jul 24) and
  `review/review/phase2s` (Jul 25) are byte-identical in every scored statistic —
  confirming they're re-scorings of the same stale pkls, not new runs. But
  `dssgd/review/review/phase2s` (Jul 27) has genuinely different numbers for NMH-2
  (`chi2` 59.78 → 120.72) and NMH-6 (`variance_decomp` values changed, plus new
  `nmh6_synthetic_control_ok` fields from this session's own A.3 fix) — real evidence of
  a later, likely-post-fix re-run. Meanwhile `ncp2_asymmetry_ratio` is `nan` in **all
  three** Phase 2 pulls, including Jul 27 — NCP-2 was never actually regenerated at all.
  So even a single Phase 2 review file can mix a fresh sub-experiment with permanently
  stale ones.
- Phase 6 showed the identical pattern one experiment over: `e12a_containment` differs
  between the Jul 24 and Jul 27 pulls (`frac_d_max_eq_G` 0/0/0 → 0.06/0.08/0.16), but
  `e12b_curvature` is byte-identical between them — E12a was re-run, E12b wasn't, in the
  same file. `e11_slopes`/`e5_crossover` are empty in both — never run for the sync track
  at all.

**Conclusion: the phase ID is not always a safe unit of provenance — different
experiments inside the same nominal "sync phase" can, and did, get regenerated at
different times relative to a mechanism fix.** A single phase-level suffix can't
capture that; only checking each experiment's own data (or, going forward, only ever
scoring data generated under the current, disambiguated protocol strings) can.

### What this changed

- `generate_queue.py`'s phase IDs are renamed `"Xs"` → `"Xsp"` throughout (matching what
  every phase branch actually and exclusively constructs: `gossip_protocol="sync_pairwise"`).
  There is no `"Xsn"` phase branch — `sync_neighbourhood` was never an intentional target
  of any phase, only a historical bug artifact — the ID is just reserved for anyone who
  wants to add it as a deliberate comparison arm later.
- Local archived data was re-triaged per the finding above, not by a blanket rename:
  - `results/phase1s` → `results/phase1sn` (pkl subdirs and filenames relabelled `SN`
    throughout — confirmed clean, un-mixed pre-fix data, no review files existed locally).
  - `results/phase2s` → `results/phase2sn` (same treatment — the raw pkls are uniformly
    the stale Jul-24 batch regardless of what the review JSONs later showed).
  - `review/phase2s` → `review/phase2sn`, `review/review/phase2s` → `review/review/phase2sn`
    (both confirmed identical-to-stale-pkls).
  - `dssgd/review/review/phase2s` — confirmed mixed (fresher NMH2/NMH6, permanently stale
    NCP2) — quarantined rather than mislabelled, under `_quarantine_mixed_mechanism_data/`.
  - `results/phase6s`, `review/phase6s`, `review/review/phase6s`,
    `dssgd/review/review/phase6s` — all quarantined under
    `_quarantine_mixed_mechanism_data/` (confirmed mixed at the E12a/E12b level).
- **Practical upshot: there is currently no valid local evidence for `sync_pairwise`'s
  E12a/E12b/E11/E5 behaviour (Phase 6) or NCP-2's asymmetry (Phase 2).** These need a
  genuine, complete re-run under the current code — ideally scored with the current
  `check_phase2.py`/`check_phase6.py` (which write explicit `"sync_pairwise"`/`"SP"`, never
  the retired `"synchronous"`/`"S"` strings) before anyone treats a result as settled.

## Two follow-on gaps this raised, and how they're addressed

Once the three mechanisms were disambiguated, two further gaps became visible that the
mechanism split alone didn't close:

### E14 — does the meritocratic filter hold across local-steps rate, for all three mechanisms?

E12/NMH-2's containment result (Thm. 4.5) had only ever been checked at one `local_steps`
value, and only for the (at the time, conflated) sync mechanism. NMH-1b's local-steps sweep
tests inter-kick relaxation timing for a single forced-flip leaf, not containment across the
generality hierarchy — it doesn't sweep `G` or `protocol`, so it can't say whether
containment survives faster/slower gossip relative to local optimisation, or whether it
differs by mechanism. `experiment_E14_meritocratic_filter_local_steps`
(`natural_cascade_experiments.py`) is a dedicated experiment for exactly this: it sweeps
`local_steps × generality level (G) × {async_poisson, sync_pairwise, sync_neighbourhood}`
and scores `cascade_depth`/`frac_d_max_eq_G` per cell
(`check_phase6.compute_e14_local_steps_table`). It is queued once, under phase `6a`
(`E14` in `generate_queue.py`), sweeping all three protocols internally rather than being
split across `6a`/`6s` — its name (`E14/G={G}/ls={ls}/proto={proto}/seed={seed}`) already
disambiguates mechanism, so it doesn't need a directory suffix the way E12a/E12b do. It is
diagnostic-only (not gated into `gate6_pass`) until there's enough real data to know what a
meaningful pass threshold looks like.

### E15 — the degenerate kick-weight law

Every protocol above the theory line applies a single fixed mixing weight `alpha` per kick
(a point mass on one value), but the paper's general reduction (Lemma 3.1's derivation, and
the r>1 hard-zero discussed for `fixation_bias`) assumes an *averaged/distributed* kick-weight
law `mu_C` — `p_+ = Pr_{C~mu_C}[C >= vartheta]`, not a single indicator. Under the repo's old
degenerate point-mass law, `fixation_bias` predicts a hard-zero cliff for `r > 1`
(`predicted_q_fix` jumps to exactly 0), while E12b's real async data shows a smooth decay
(`0.16 → 0.14 → 0.10 → 0.06 → 0.04`) — a qualitative mismatch that was previously only
noted as a documented theory-vs-model tension, not tested against an alternative model.

Two additions close this gap without touching the existing degenerate-law code path:

- `theory.fixation_bias_distributed(a, b, delta_norm, r, kick_weight_cdf)` — the same
  birth-death bias `rho = p_-/p_+`, but `p_+`/`p_-` are computed from a supplied kick-weight
  CDF (default `uniform_kick_cdf`, i.e. `Uniform(0,1)`) instead of a single indicator
  threshold. Numerically this gives a smooth decay (`q_fix ≈ 0.25 → 0.19` over `r=1..2`)
  instead of the old hard cliff, matching the *shape* of the real E12b async discrepancy.
  `fixation_bias` itself is deliberately left unchanged — the two functions model different
  laws, not competing bugfixes of the same one.
- `AsynchronousGossip`/`SynchronousPairwiseGossip` gained an optional
  `alpha_sampler: Callable[[], float]` constructor argument (via
  `make_uniform_alpha_sampler`); when set, each kick draws its mixing weight fresh instead of
  using the fixed `alpha`. `GossipAveraging` (`sync_neighbourhood`) has no scalar alpha to
  distribute and rejects `kick_weight_law != "fixed"` outright
  (`clique_fixation.run_clique_fixation_trial`).
- `experiment_E15_distributed_kick_curvature_ratchet` (`clique_fixation.py`) is E12b's
  curvature-ratchet sweep re-run with `kick_weight_law="uniform"` instead of `"fixed"` — a
  new experiment, not a reinterpretation of E12b's existing outputs, queued separately under
  `6a`/`6s` (`E15`/`E15SP`) and scored by `check_phase6.compute_e15_curvature_table` against
  `fixation_bias_distributed` rather than `fixation_bias`. Also diagnostic-only for now.
