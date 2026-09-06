# Mechanistic attribution of Relational Transformer behavior to synthetic pretraining generators

Four Relational Transformer checkpoints share an architecture, an initialization and a
step budget, and differ only in which synthetic relational-database generator produced
their pretraining corpus. That makes the generator the single experimental variable, so
any mechanistic difference between the four models is attributable to the data.

This repository holds the analysis code, the results and the paper. It asks what each
generator actually taught its model, and whether the thing it taught is what produces the
downstream benchmark ranking.

The four arms are RelDiff and GRDM, which regenerate a real reference database's schema
and foreign-key graph with a diffusion model, and PluRel and RDB-PFN, which sample both
schema and values from a hand-designed prior with no reference data at all.

## What the analysis found

The corpus property that matters is measurable in the data before any model exists.
Fitting gradient-boosted predictors to recover a held-out cell from its own row, then
from its row plus its foreign-key parents, RelDiff is the only corpus where neighbours
measurably help (median gain 0.072, 95% CI [0.018, 0.091]). RDB-PFN's interval spans
zero. Roughly 77% of RelDiff's predictable signal comes from neighbours, against 0.1% for
RDB-PFN.

That property predicts the mechanism. RelDiff is the only arm whose weights deviate most
in the neighbour-attention stream, the only one whose loss degrades under foreign-key
corruption, and the only one whose benchmark advantage disappears when that mechanism is
ablated.

Two published claims did not survive re-measurement, and the corrections are in the
paper:

The claim that RelDiff is the only model whose accuracy improves with longer context was
an artifact of sample size. The two tasks carrying it had six and ten positive examples
at 30k context. Rescored at n=2048 (about 60 positives), every model degrades and RelDiff
simply degrades least, by a factor of three or more. One ablation cell changed sign
entirely: GRDM on user-engagement went from -0.163 to +0.102.

The description of `nbr` as the only stream carrying cross-table information is wrong.
Intervening on each route separately shows the two are in series, not parallel: `nbr`
lets a parent aggregate its children, but that aggregate reaches a masked cell only
through `feat`'s parent-cell inclusion. Blocking `feat` costs RelDiff 0.193 and
additionally zeroing `nbr` costs nothing further, while zeroing `nbr` alone costs 0.132.

Three controls turned out to matter more than expected. An untrained network already
degrades 4.9% under foreign-key shuffling, simply because changing the mask changes the
computation; against that floor the three structure-blind arms sit at 0.7 to 1.3x, so
their indifference is not a small effect but the absence of one, while RelDiff reaches
17x. Corrupting a graded fraction of links gives a monotone curve for RelDiff at 11 to
29x the others' slope. And knocking out individual neighbour-attention heads shows only
RelDiff has load-bearing ones, with a single head carrying 40% of the head-attributable
cost against a uniform expectation of 12.5%.

One thing worth flagging for anyone reusing this setup: GRDM's shipped corpus contains no
text or datetime columns at all. Every string was converted to a float and never
converted back, so the RT's text encoder and MiniLM pathway received no signal in that
arm. The GRDM arm therefore differs from the others in more than its generator.

## Layout

```
paper/                 LaTeX source, figures and the compiled PDF
analysis/              experiment scripts, one per experiment
analysis/out/          every result as CSV, plus figures
analysis/RESULTS.md    narrative writeup of each run
rt_icl/                the RT pipeline, recovered from the notebooks
RT_Benchmark.ipynb     benchmark harness (Colab)
RT_Benchmark_Prep.ipynb  evaluation-data preparation (Colab)
rt_benchmark.csv       the benchmark sweep this analysis builds on
diversity_rfms.csv     generator output diversity
```

`rt_icl/` was extracted from cell 5 of the two notebooks, where it ships as a
gzip+base64 payload. It is unmodified; `analysis/` imports it rather than reimplementing
anything.

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

GPU, and the RT stack. These need `rt_icl` on the path, the plurel repository for `rt/`
and `rustler`, relbench 2.1.2, and a prepared evaluation database. An Ampere-or-newer card
is required: the checkpoints are cast to bfloat16 and FlexAttention's kernels need compute
capability 8.0 or above.

```bash
python analysis/e7_reliance_dose.py      # reliance, dose-response, random-init floor
python analysis/e45_paths_and_heads.py   # route separation and head knockout
python analysis/x1_repower_ablations.py  # benchmark under ablation at honest n
python analysis/e6_linear_probes.py      # linear probes (see the caveat below)
python analysis/e3_transfer_matrix.py    # cross-generator transfer (needs the corpora)
```

Then `python analysis/make_paper_figures.py` regenerates the figures.

Checkpoints and corpora are not in the repository. Put the four `*_final.pt` files in
`model_checkpoints/` and the corpus archives in `data/`, or pass `--ckpt-dir` and the
corpus paths explicitly.

## A negative result worth knowing about

`e6_linear_probes.py` does not discriminate between the arms, and the reason is
structural rather than a bug. The RT is a residual stack, so the input embedding stays
linearly present in the residual stream at every depth. Any probe target that is a
function of a cell's own inputs is therefore decodable by a linear map without the network
having learned anything, which is why an untrained model scores 1.000 on parent-table
identity. Row degree and parent-table identity are both read off `f2p_nbr_idxs`, an input.

A probe that would discriminate has to target something requiring computation, such as a
masked cell's parents' values. The E6 plan in `MECH_INTERP_EXPERIMENTS.md` has this flaw
for probes (a), (b) and (e) as written.

## A methodological note

Weight-space measurements failed to predict their functional counterparts three separate
times here. E1's clustering inverted under CKA. Per-head weight concentration said all
four models were equally uniform, but knocking heads out behaviourally showed only RelDiff
has load-bearing ones. And an encoder deviation that looked large per parameter turned out
to be the text pathway alone, with scalar featurization effectively frozen.

Weight-space structure is a hypothesis about function, not evidence of it. Every claim in
the paper that rests on weights has a behavioural counterpart for that reason.

## Status

E3, the cross-generator transfer matrix, is written but unrun: it needs the four
preprocessed corpora, which were not retrievable at the time. Everything else in
`analysis/` has been run and its output is in `analysis/out/`.
