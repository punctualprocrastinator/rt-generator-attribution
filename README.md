# Mechanistic attribution of Relational Transformer behavior to synthetic pretraining generators

Four Relational Transformer checkpoints share an architecture, an initialization and a
step budget, and differ only in which synthetic relational-database generator produced
their pretraining corpus. That makes the generator the single experimental variable, so
any mechanistic difference between the four models is attributable to the data.

This repository holds the analysis code, every result, and the paper. It asks what each
generator actually taught its model, and whether the thing it taught is what produces the
downstream benchmark ranking.

The four arms are RelDiff and GRDM, which regenerate a real reference database's schema
and foreign-key graph with a diffusion model, and PluRel and RDB-PFN, which sample both
schema and values from a hand-designed prior with no reference data.

Architecture is identical across arms: 12 blocks, d=256, 19.2M parameters, three attention
streams per block (`col`, `feat`, `nbr`), typed value encoders, masked-cell reconstruction
objective. The shared initialization is verifiable rather than assumed: 132 tensors
(115,585 parameters) are bitwise identical across all four checkpoints, which independently
trained models cannot achieve by coincidence.

## Results

### The corpus property, measured before any model exists

Gradient-boosted predictors recover a held-out cell from its own row, then from its row
plus its foreign-key parents, then plus aggregates of its children. The gain is the
neighbour predictive information in that corpus. 40 databases each, medians with 95%
bootstrap intervals, no RT involved.

| corpus | tasks | own-row R2 | parent lift | child lift |
|---|---|---|---|---|
| RelDiff | 48 | 0.022 | **+0.072 [+0.018, +0.091]** | -0.001 |
| PluRel | 113 | 0.166 | +0.009 [+0.001, +0.022] | -0.000 |
| RDB-PFN | 120 | 0.436 | +0.001 [-0.004, +0.009] | -0.005 |
| GRDM | 120 | -0.011 | -0.057 [-0.076, -0.044] | -0.105 |

RelDiff is the only corpus where a neighbouring table measurably helps. Roughly 77% of its
predictable signal comes from neighbours, against 5% for PluRel and 0.1% for RDB-PFN.
RDB-PFN is the mirror image: the most learnable within-row structure of any corpus and no
relational signal at all.

Child lift is flat everywhere, RelDiff included, so what its data rewards specifically is
reading parents.

### Weight-space fingerprint

Deviation from the four-model mean, by stream. RelDiff is the only arm peaking in `nbr`;
GRDM peaks in `feat`.

| model | encoders | col-attn | feat-attn | nbr-attn | ffn |
|---|---|---|---|---|---|
| GRDM | 1.95 | 6.10 | **7.18** | 5.55 | 10.83 |
| PluRel | 1.05 | 5.13 | 4.68 | 3.90 | 7.69 |
| RDB-PFN | 1.03 | 5.57 | 4.35 | 3.80 | 7.51 |
| RelDiff | 1.64 | 5.37 | 5.88 | **6.43** | 9.49 |

Typed featurization turns out to be data-insensitive: the number, datetime and boolean
encoders agree across arms to a cosine of 0.99998, and the mask embeddings to 1.000000.
The entire encoder deviation above is the text and col_name projections, which are 98.5% of
that parameter group.

### Structure reliance, against an untrained floor

Masked-cell loss under corruption, rel-stack. The floor matters: an untrained network
already degrades 4.9% under foreign-key shuffling, simply because changing the mask changes
the computation.

| model | intact | relative degradation under FK shuffle | ratio to untrained floor |
|---|---|---|---|
| RelDiff | 0.241 | **83.6%** | **17.1x** |
| PluRel | 0.161 | 6.1% | 1.26x |
| GRDM | 0.342 | 5.3% | 1.08x |
| RDB-PFN | 0.208 | 3.4% | 0.69x |
| untrained | 0.556 | 4.9% | 1.00x |

The three structure-blind arms sit at 0.7 to 1.3x the floor. Their indifference is not a
small effect but the absence of one.

Corrupting a graded fraction of links rather than all of them gives a monotone curve for
RelDiff at 11 to 29x the others' slope, which noise does not produce:

| model | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| RelDiff | 0.241 | 0.315 | 0.372 | 0.418 | 0.443 |
| GRDM | 0.342 | 0.350 | 0.352 | 0.359 | 0.360 |
| PluRel | 0.161 | 0.164 | 0.166 | 0.170 | 0.171 |
| RDB-PFN | 0.208 | 0.210 | 0.211 | 0.213 | 0.215 |

### The two cross-table routes are in series

The RT can move information across a foreign-key edge two ways, and both are built from the
same `f2p_nbr_idxs` tensor, which is why corrupting that input cannot tell them apart.
Intervening on each separately:

| condition | RelDiff loss | cost |
|---|---|---|
| intact | 0.280 | |
| `nbr` zeroed | 0.411 | +0.132 |
| `feat` parent path blocked | 0.473 | +0.193 |
| both | 0.473 | **+0.193, identical** |

`nbr` lets a parent aggregate its children, but that aggregate reaches a masked cell only
through `feat`'s parent-cell inclusion. Block the second hop and the first becomes
unreachable, so zeroing `nbr` on top costs exactly nothing. The `nbr` stream still produces
output under the block (mean magnitude unchanged at 0.077), so it is not being silenced.

`feat` is the bottleneck. Describing `nbr` as the only stream carrying cross-table
information, as the pretraining literature and an earlier draft of this paper did, is wrong.

### Benchmark, and a claim that did not survive re-measurement

The published claim that RelDiff is the only model whose accuracy improves with longer
context was an artifact of sample size. The two tasks carrying it had six and ten positive
examples at 30k context. Rescored at n=2048 (about 60 positives):

| model | @1024 | @30k, n=256 | @30k, n=2048 | delta, published | delta, corrected |
|---|---|---|---|---|---|
| RelDiff | 0.748 | 0.754 | 0.737 | +0.006 | **-0.011** |
| GRDM | 0.666 | 0.627 | 0.628 | -0.040 | -0.038 |
| PluRel | 0.723 | 0.667 | 0.681 | -0.057 | -0.043 |
| RDB-PFN | 0.740 | 0.653 | 0.670 | -0.087 | -0.069 |

Every model degrades with context. RelDiff degrades least, by a factor of three or more.
The shifts follow the small-sample signature exactly: extreme estimates regress toward the
middle, and every already-well-powered cell moves by at most 0.007.

Ablating the mechanism at 30k, all cells at n=2048:

| model | intact | FK shuffled | nbr zeroed |
|---|---|---|---|
| RelDiff | **0.865** | **0.579** | **0.624** |
| PluRel | 0.652 | 0.650 | 0.636 |
| RDB-PFN | 0.642 | 0.643 | 0.634 |
| GRDM | 0.520 | 0.622 | 0.537 |

Corrupting foreign-key structure costs RelDiff 0.286 AUROC, dropping it below the level the
structure-blind models reach untouched. One cell changed sign against the underpowered
version: GRDM went from -0.163 to +0.102, so corruption *helps* it. That is the
misleading-priors effect, and it means the published table asserted the opposite of what
holds there.

### Head knockout

Removing one `nbr` head at a time, across all 12 blocks:

| model | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 |
|---|---|---|---|---|---|---|---|---|
| RelDiff | **+0.032** | +0.001 | +0.015 | +0.011 | +0.010 | -0.003 | +0.006 | +0.006 |
| GRDM | -0.002 | +0.000 | -0.002 | +0.001 | +0.000 | -0.002 | +0.002 | -0.001 |
| PluRel | -0.000 | +0.000 | -0.001 | -0.000 | +0.001 | -0.002 | +0.001 | -0.000 |
| RDB-PFN | -0.001 | +0.001 | -0.000 | +0.000 | +0.001 | +0.000 | -0.000 | +0.001 |

Only RelDiff has load-bearing neighbour heads at all. Within it, head 0 carries 40% of the
head-attributable cost against a uniform expectation of 12.5%. It is not a single critical
head though: removing head 0 costs 0.032 while removing the whole stream costs 0.128, and
the individual removals sum to less than the whole, so the heads are redundant and
super-additive.

### A confound worth knowing about

GRDM's shipped corpus contains no text or datetime columns at all. Every string was
converted to a float and never converted back: 0 object columns against RelDiff's 80 on
rel-f1 and 252 on rel-trial, over the same reference databases.

| corpus | numeric | text or categorical | boolean |
|---|---|---|---|
| RelDiff | 512 | 234 | 0 |
| PluRel | 1132 | 0 | 518 |
| RDB-PFN | 1081 | 0 | 0 |
| GRDM | 518 | 0 | 0 |

The RT has a dedicated text encoder and a 384-dimension MiniLM pathway, which in the GRDM
arm received no training signal. That arm therefore differs from the others in more than
its generator, and its last-place finish has a simpler explanation available than its
generator design.

## A negative result

`e6_linear_probes.py` does not discriminate between the arms, and the reason is structural
rather than a bug. The RT is a residual stack, so the input embedding stays linearly present
in the residual stream at every depth. Any probe target that is a function of a cell's own
inputs is therefore decodable by a linear map without the network having learned anything,
which is why an untrained model scores 1.000 on parent-table identity. Row degree and
parent-table identity are both read off `f2p_nbr_idxs`, an input.

| probe (block 11) | RelDiff | GRDM | PluRel | RDB-PFN | untrained |
|---|---|---|---|---|---|
| semantic type | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| table identity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| parent table | 0.9995 | 0.9998 | 0.9998 | 0.9994 | 1.0000 |
| row degree (R2) | 0.935 | 0.937 | 0.923 | 0.920 | 0.924 |

A probe that would discriminate has to target something requiring computation, such as a
masked cell's parents' values.

## A methodological note

Weight-space measurements failed to predict their functional counterparts three separate
times here. The weight-space clustering inverted under CKA on real data. Per-head weight
concentration said all four models were equally uniform (Gini 0.016 to 0.020), but knocking
heads out behaviourally showed only RelDiff has load-bearing ones. And an encoder deviation
that looked large per parameter turned out to be the text pathway alone.

Weight-space structure is a hypothesis about function, not evidence of it. Every claim in
the paper that rests on weights has a behavioural counterpart for that reason.

## Layout

```
paper/                 LaTeX source, figures and the compiled PDF
analysis/              experiment scripts, one per experiment
analysis/out/          every result as CSV, plus figures
rt_icl/                the RT pipeline, recovered from the notebooks
RT_Benchmark.ipynb     benchmark harness (Colab)
RT_Benchmark_Prep.ipynb  evaluation-data preparation (Colab)
rt_benchmark.csv       the benchmark sweep this analysis builds on
diversity_rfms.csv     generator output diversity
```

`rt_icl/` was extracted from cell 5 of the two notebooks, where it ships as a gzip+base64
payload. It is unmodified; `analysis/` imports it rather than reimplementing anything.

## Reproducing

Scripts split by what they need.

Weights only, no GPU and no data, a few minutes on a laptop:

```bash
python analysis/e1_weight_divergence.py       # weight-space divergence, shared-init check
python analysis/e2enc_encoder_geometry.py     # typed-encoder and mask-embedding geometry
python analysis/e8w_head_concentration.py     # per-head weight concentration
```

Corpora only, CPU, tens of minutes:

```bash
python analysis/e2_predictive_necessity.py --dbs 40
python analysis/e2_aggregate.py
```

GPU, and the RT stack. These need `rt_icl` on the path, the plurel repository for `rt/` and
`rustler`, relbench 2.1.2, and a prepared evaluation database. An Ampere-or-newer card is
required: the checkpoints are cast to bfloat16 and FlexAttention's kernels need compute
capability 8.0 or above.

```bash
python analysis/e7_reliance_dose.py      # reliance, dose-response, untrained floor
python analysis/e45_paths_and_heads.py   # route separation and head knockout
python analysis/x1_repower_ablations.py  # benchmark under ablation at honest n
python analysis/e6_linear_probes.py      # linear probes (see the negative result above)
python analysis/e3_transfer_matrix.py    # cross-generator transfer (needs the corpora)
```

Then `python analysis/make_paper_figures.py` regenerates the figures.

Checkpoints and corpora are not in the repository. Put the four `*_final.pt` files in
`model_checkpoints/` and the corpus archives in `data/`, or pass `--ckpt-dir` and the corpus
paths explicitly.

## Status

E3, the cross-generator transfer matrix, is written but unrun: it needs the four
preprocessed corpora, which were not retrievable at the time. Everything else in `analysis/`
has been run and its output is in `analysis/out/`.
