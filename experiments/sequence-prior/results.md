# E48 — Count-based history sequence prior (competitor-borrowed)

**Group:** `research/sequence-prior` · **Status:** ✅ DONE — ❌ REJECTED (net −0.0009, gate fail)

Screen: [seq_prior_screen.py](seq_prior_screen.py) · leak-free fold-0 14k val · base = champion AWP **t031** (LB 0.79300)

## Origin

Borrowed from the youngyoung2345 competitor repo (mmBERT ensemble, Public 0.7807). Their
one *shipped* post-hoc correction was a count-based prior over the 14 actions keyed on the
ordered sequence of prior assistant-action names in a row's history (optionally + the last
action's result status), added to the model logits and applied to `sess_sim_` rows only,
support-gated. It is **not** a learned combiner (those are dead — [[uniform-mean-unbeatable]]);
it is a fixed non-neural prior, so it is not covered by the E29 combiner shoot-out.

## Design

Prior built from the 56k fold-0 **train** rows only, applied to the 14k fold-0 **val** rows,
scored against the same champion AWP model (t031) whose fold-56k checkpoint never saw the 14k
→ genuinely leak-free. `argmax( log p_model + w · log_prior[key] )`, add-1 (Laplace) smoothed,
SIM-rows-only, AU / history-0 / under-supported rows kept **bit-exact**. Swept key mode
(`actions` vs `actions+[R_status]`), history depth (full / last-3 / last-2 / last-1),
min-session-support ∈ {1,3,5,10}, weight ∈ {0.3,0.6,1.0,1.5} — 128 configs.

## Result

| metric (best config: actions·full·supp10·w0.6) | value |
|---|---|
| baseline macro-F1 (t031, 14k) | 0.7806 |
| Δ macro-F1 full | **−0.0009** |
| Δ macro-F1 SIM | −0.0009 |
| Δ macro-F1 finding sub-family | +0.0004 |
| corrections / regressions | 77 / 78 |
| AU rows changed | 0 (routing correct) |

**Every one of the 128 configs is net-negative on full and SIM macro-F1.** Corrections and
regressions cancel (~77/78 at the best point); the only non-negative signal is a noise-level
+0.0004–0.0006 on the 4-class finding sub-family, far below the +0.001 gate and inside LB noise.
No config clears the preregistered gate (full ≥ +0.001 ∧ sim ≥ +0.001 ∧ finding ≥ 0 ∧
corrections > regressions ∧ AU untouched).

**Robustness across baseline strength** (best config, applied to all four banked AWP fold models):

| base | baseline F1 | Δ full | corr/regr |
|---|---|---|---|
| t001 | 0.7795 | −0.0003 | 70/65 |
| t040 | 0.7803 | +0.0003 | 86/77 |
| t070 | 0.7805 | −0.0008 | 82/84 |
| t031 | 0.7806 | −0.0009 | 90/88 |

All four strong AWP baselines sit at ≈0 Δ with corrections ≈ regressions — the prior carries
no information these models don't already hold.

## Verdict

❌ **Rejected on our champion.** The history→next-action signal the count prior encodes is
already captured by the AWP+LS encoder; adding it externally just trades corrections for
equal-sized regressions. This mirrors the recurring pattern that a strong single absorbs
structured side-signals (LS absorbs label noise E22; uniform mean unbeats learned combiners
E29; E27 flip-rule null). The competitor's +gain from this prior is a **headroom artifact of
their weaker 0.7807 mmBERT baseline**, not a transferable accuracy lever. Not worth the
inference-time key machinery or a submission slot. The only faint positive (finding sub-family)
is below LB noise. Method line closed; no further prior/residual borrows from that repo planned
(their frozen residuals + session-random-effect were already exploratory/rejected on their side).
