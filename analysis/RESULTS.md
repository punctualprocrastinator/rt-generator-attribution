# Results from the local runs

Everything here was produced on CPU from the checkpoints and corpora already in this
repository. Scripts are in `analysis/`, outputs and figures in `analysis/out/`.
Nothing here needed the GPU box.

## 1. E1 reproduces, and shared initialization is now confirmed

`analysis/e1_weight_divergence.py` recomputes the weight-space table from the four
checkpoints. It matches the published table cell for cell, with one rounding
difference (GRDM's FFN norm is 10.83 against the paper's 10.82).

| model | encoders | col-attn | feat-attn | nbr-attn | ffn |
|---|---|---|---|---|---|
| grdm | 1.95 | 6.10 | 7.18 | 5.55 | 10.83 |
| plurel | 1.05 | 5.13 | 4.68 | 3.90 | 7.69 |
| rdbpfn | 1.03 | 5.57 | 4.35 | 3.80 | 7.51 |
| reldiff | 1.64 | 5.37 | 5.88 | 6.43 | 9.49 |

Total deviation norms (grdm 15.51, reldiff 14.05, plurel 11.12, rdbpfn 11.04) and
deviation cosines (plurel~rdbpfn +0.113, grdm~plurel -0.458, grdm~rdbpfn -0.462) also
match. RelDiff's peak in `nbr` and GRDM's in `feat` are both real.

The open question from `MECH_INTERP_EXPERIMENTS.md` is settled. **132 tensors are
bitwise identical across all four checkpoints**, covering 115,585 parameters, mostly
decoder heads, type norms and several mask embeddings. Four independently trained
models cannot arrive at bit-identical tensors by coincidence, so the arms provably
share an initialization and those parameters received no gradient. The paper can state
this rather than infer it from cosines near 0.99.

One caveat the paper should absorb. Raw L2 norms scale with tensor size, which is why
FFN looks largest and the encoders look untouched. Per parameter the ordering changes:

| model | encoders | col | feat | nbr | ffn |
|---|---|---|---|---|---|
| grdm | 4.386 | 3.440 | 4.047 | 3.129 | 3.525 |
| plurel | 2.345 | 2.893 | 2.636 | 2.198 | 2.504 |
| rdbpfn | 2.311 | 3.142 | 2.455 | 2.142 | 2.445 |
| reldiff | 3.690 | 3.025 | 3.315 | 3.622 | 3.089 |

(RMS deviation, x1e-3.) The encoders are the *most* moved stream per parameter for both
real-reference generators, so the claim that featurization is data-insensitive does not
survive normalization. The central claim does survive: among the three attention
streams RelDiff still peaks in `nbr` (3.622, above feat 3.315 and col 3.025) and GRDM
still peaks in `feat` (4.047). Either report the normalized table or drop the encoder
claim.

Figures: `fig_e1_stream_depth.png`, `fig_e1_stream_bars.png`.

## 2. GRDM's corpus has no text and no dates

Column type census over the shipped corpora, counting columns across sampled databases:

| corpus | numeric | text or categorical | boolean |
|---|---|---|---|
| reldiff | 512 | 234 | 0 |
| plurel | 1132 | 0 | 518 |
| rdbpfn | 1081 | 0 | 0 |
| grdm | 518 | 0 | 0 |

On the same reference database, RelDiff ships 80 string columns for rel-f1 and 252 for
rel-trial. GRDM ships zero in both. Every string and every date became a float and was
never converted back, which is the failure mode the notes already predicted for GRDM.
This is also independent evidence that the unlabeled `synthetic-2026` zip is the GRDM
corpus, since it fails in GRDM's documented way.

It matters for the controlled comparison. The RT has typed encoders and a 384-dimension
MiniLM text pathway, and in the GRDM arm that pathway saw no training signal at all. So
the GRDM arm differs from the other three in more than its generator, and a simpler
explanation is available for its last place finish (mean AUROC 0.666) than the
feat-attention account the paper gives. It also explains why GRDM's encoders moved most
per parameter in E1.

## 3. Predictive necessity, measured

`analysis/e2_predictive_necessity.py` and `e2_aggregate.py`. For each target column,
LightGBM predicts a held-out cell three times: from its own row, from its own row plus
parent columns joined over the foreign key, and from its own row plus aggregates of its
children. No RT is involved. 40 databases per corpus, 48 to 120 tasks each, scores
clipped to [-1, 1] before differencing because unbounded negative R2 lets one diverged
fit swamp a mean. Medians with 95% bootstrap intervals:

| corpus | tasks | own-row R2 | parent lift | child lift |
|---|---|---|---|---|
| reldiff | 48 | 0.022 | **+0.072 [+0.018, +0.091]** | -0.001 |
| plurel | 113 | 0.166 | +0.009 [+0.001, +0.022] | -0.000 |
| rdbpfn | 120 | 0.436 | +0.001 [-0.004, +0.009] | -0.005 |
| grdm | 120 | -0.011 | -0.057 [-0.076, -0.044] | -0.105 |

RelDiff is the only corpus where reading a neighboring table measurably helps recover a
held-out cell. Its parent lift is eight times PluRel's and about seventy times
RDB-PFN's, whose interval spans zero.

The contrast is sharper as a share. Taking the median own-row R2 and the median parent
lift as the two sources of predictable signal, the fraction contributed by neighbors is
77% for RelDiff, 5% for PluRel and 0.1% for RDB-PFN. RelDiff's cells are close to
unrecoverable from their own row (R2 0.022) and become recoverable once a parent is
visible, which is exactly the predictive necessity the paper describes. RDB-PFN is the
mirror image: the most learnable within-row structure of any corpus (0.436) and no
relational signal at all, matching its profile of good short-context attribute transfer
and no structure use. Parents help on 52% of RelDiff tasks against 22% for RDB-PFN.

This closes the open end of the causal chain. The corpus node was previously a
description of how each generator was built, and it is now a measured quantity that
orders the generators the same way the weights (E1), the ablations (E7) and the
benchmark (X1) do.

Two honest qualifications.

Child lift is flat for every corpus including RelDiff. No corpus rewards aggregating
children, so what RelDiff's data teaches is specifically "read your parents". The paper
leans on child-aggregation tasks as RelDiff's signature wins, so either that skill is
transfer rather than something the corpus directly incentivizes, or mean, standard
deviation and count are too crude to capture the child signal. Worth one more pass with
richer aggregates before leaning on it.

GRDM's parent lift is significantly negative and its own-row R2 is zero, meaning its
numeric cells are close to unpredictable from anything. That is consistent with the type
destruction in section 2 rather than with a statement about GRDM's relational design.

Figure: `fig_e2_lift.png`.

## 4. The benchmark re-power (deadline-critical fix)

`rt_icl` was recovered from cell 5 of the RT_Benchmark notebooks (a gzip+base64 payload)
and the full pipeline was rebuilt on the GPU box. The two rare-label tasks were rescored
at 30k context with `max_samples=2048` instead of 256, taking them from 6 and 10 positive
examples to about 60.

The published claim that RelDiff is the only model whose accuracy improves with context
does not survive. Its inflated estimates fall (user-engagement 0.929 to 0.865, user-badge
0.847 to 0.812) while the pooling models' deflated ones rise (PluRel 0.564 to 0.652,
RDB-PFN 0.538 to 0.642), and every already-well-powered cell moves by at most 0.007 --
the signature of small-sample regression. Corrected, all four models degrade with
context and RelDiff simply degrades least, by a factor of three or more: -0.011 against
-0.038, -0.043 and -0.069.

The X1 ablations were rerun at the same sample size (24 cells). RelDiff loses 0.286 AUROC
to FK corruption at 30k, falling below the level the structure-blind models reach
untouched, while PluRel and RDB-PFN move by at most 0.017. One sign flipped: GRDM on
user-engagement went from -0.163 at n=256 to +0.102 at n=2048, so the published table
asserted the opposite of what holds for that cell.

## 5. Controls that sharpen the reliance result

An untrained RT degrades 4.9% under FK shuffling, purely because altering the mask alters
the computation. Against that floor the three structure-blind arms sit at 0.7 to 1.3x and
RelDiff at 17x. Their indifference is not a small effect but the absence of one.
Corrupting a graded fraction of links gives a monotone curve for RelDiff at 11 to 29x the
others' slope.

## 6. The two cross-table routes are in series

Intervening on each route separately (blocking `feat`'s parent-cell term, zeroing `nbr`,
both) shows they are not parallel. `nbr` lets a parent aggregate its children, but that
aggregate reaches a masked cell only through `feat`. Blocking `feat` costs RelDiff +0.193
and additionally zeroing `nbr` costs nothing further (identical to six decimals), while
zeroing `nbr` alone costs +0.132. The `nbr` stream still produces output under the `feat`
block (mean magnitude unchanged at 0.077), so it is not silenced -- its product is simply
unreachable. The paper's description of `nbr` as the only cross-table stream was wrong.

## 7. Weights and function disagree, repeatedly

Three times now a weight-space measure has failed to predict its functional counterpart.
E1's clustering inverted under CKA. The per-head weight Gini said all four models were
equally uniform (0.016 to 0.020), but knocking heads out behaviourally shows only RelDiff
has load-bearing `nbr` heads at all, with head 0 carrying 40% of the head-attributable
cost against a uniform 12.5%. And the encoder deviation that looked large per parameter
turned out to be the text pathway alone, with scalar featurization frozen. Weight-space
concentration is not evidence about function; it is a hypothesis to test.

## Negative result: E6 as specified does not discriminate

Linear probes for semantic type, table identity, parent-table identity and row degree all
sit at ceiling for every arm including an untrained one (gains over random init below
0.014, negative for two models). The cause is structural: the RT is a residual stack, so
the input embedding remains linearly present at every depth, and any probe target that is
a function of a cell's own inputs is decodable without the network having learned
anything. `parent_table` and `degree` are both read off `f2p_nbr_idxs`, an input. A probe
that discriminates would need to target something requiring computation, such as a
masked cell's parents' values. The E6 plan in `MECH_INTERP_EXPERIMENTS.md` shares this
flaw for probes (a), (b) and (e).

## Still blocked

E3, the cross-generator transfer matrix, needs the four `*_pre.tar` corpora, which Google
Drive refused on both folder and per-file paths (rate limiting after a large pull, plus
narrower per-file permissions). `analysis/e3_transfer_matrix.py` is written and will run
unmodified once they are reachable.
