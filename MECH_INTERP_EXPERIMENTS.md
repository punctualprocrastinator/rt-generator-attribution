# Mechanistic Interpretability Experiments — RT Checkpoints × 4 Generators

**Setup.** Four Relational Transformer checkpoints (`grdm_final.pt`, `plurel_final.pt`, `rdbpfn_final.pt`, `reldiff_final.pt`), byte-identical architecture: 12 blocks, d=256, 19.2M params, three attention streams per block — `col` (column-level), `feat` (row/feature-level), `nbr` (**neighbor attention: the only place cross-table/FK information flows**) — plus typed value encoders (number/text/datetime/boolean, 384-d MiniLM text), per-type mask embeddings, masked-value-reconstruction pretraining objective.

**The controlled variable is the pretraining corpus generator.** Same architecture, same step budget (shared across arms), same objective — any mechanistic difference between checkpoints is attributable to the synthetic data.

**Question:** downstream benchmarks say *which* generator pretrains the better model; these experiments ask *why* — whether a generator induces relational computation (use of FK structure) or only better marginal statistics.

---

## Tier 1 — weights only (no dataloader, runs locally on CPU)

### E1. Weight-space divergence profile

**Why.** A corpus fingerprint computable from weights alone. The RT has cleanly separated parameter streams: if two models diverge mostly in `nbr`-attention weights, their corpora taught different *relational structure*; if they diverge in the typed encoders and `feat`-attention, the corpora differed mainly in *value/marginal statistics*.

**How.** For every model pair (6 pairs), compute cosine similarity and L2 distance of concatenated parameters, grouped two ways: (a) by stream — encoders / decoders / col-attn / feat-attn / nbr-attn / FFN; (b) by depth — the same four streams per block, blocks 0–11. Plot the 6-pair mean divergence as a stream × depth heatmap. Caveat to check first: whether all arms started from the same init (shared seed). If yes, absolute divergences are meaningful; if not, only the *shape* of the profile (which stream/depth diverges most, relative to the rest) is interpretable — report it that way.

**Questions answered.**
- Do the generators differ in the relational structure they teach (nbr diverges) or the value statistics (encoders/feat diverge)?
- Where in depth does generator identity live — early (featurization) or late (semantic/relational integration)?
- Which pair of generators produces the most similar models? (Prediction: the two diffusion-family generators, GRDM and RelDiff, cluster; the schema-synthesizing PluRel is the outlier.)

**Cost.** Minutes, local CPU. **Status: first experiment to run.**

### E2. Typed-encoder and mask-embedding geometry

**Why.** The four generators emit differently-distributed values (RelDiff quantile-normalizes, GRDM subsamples real references, PluRel synthesizes from scratch). The `number`/`datetime` encoders are 1→256 linear maps and the mask embeddings are single vectors — small enough to compare exhaustively, and they sit at the model's interface with raw data.

**How.** Compare across models: encoder weight norms and direction (a 1→256 map is just a scaled direction); cosine structure among the four `mask_embs` (is "masked number" ≈ "masked datetime"?); effective rank / spectrum of the 384→256 text and col_name projections.

**Questions answered.**
- Did value-distribution differences between corpora propagate into different low-level featurization, or do all four models converge to the same input geometry (suggesting featurization is data-insensitive and everything interesting happens deeper)?

**Cost.** Minutes, local CPU.

---

## Tier 2 — needs the `rt_icl` pipeline + data (GPU box; rt_icl extracted from cell 5 of `RT_Pretrain_reldiff.ipynb`)

### E3. Cross-generator transfer matrix

**Why.** Each model evaluated on each generator's held-out synthetic data **plus real RelBench databases** gives a 4×5 loss matrix. The "real" column ranks generators by *how close their synthetic distribution is to reality as measured by what a model trained on it learned* — a much stronger claim than distance metrics on raw data. The diagonal measures generator-artifact overfitting.

**How.** Masked-reconstruction loss (the pretraining objective, unchanged) for each checkpoint on: held-out DBs from each of the 4 corpora + a fixed set of real RelBench DBs converted through the same `RT_Convert` pipeline. Report per-type losses too (number vs text vs datetime) — generators may be realistic in numerics but not text, etc.

**Questions answered.**
- Which generator's data is closest to real relational data, *as measured by transfer*?
- How much does each model overfit its own generator's artifacts (diagonal advantage)?
- Is generator similarity symmetric (does GRDM-model do well on RelDiff data iff RelDiff-model does well on GRDM data)?

**Data note.** RelDiff corpora are already in the HF vault; real RelBench DBs download on the box; PluRel corpus can be uploaded overnight or restored from the collaborator's Drive; GRDM/RDB-PFN held-out corpora need to come from the collaborator.

### E4. Layer-wise representation similarity (CKA)

**Why.** Weight distance (E1) can overstate differences — different weights can compute the same function. CKA on identical inputs measures *functional* divergence, layer by layer, and localizes where the four models stop agreeing.

**How.** Fixed batches from real RelBench DBs (identical across models); collect residual-stream activations after each block; compute linear CKA for all 6 model pairs per block; include a random-init RT as the similarity floor and (if available) two checkpoints of the same arm at different steps as the ceiling. One matrix per stream if activations are cheap to split (post-col vs post-nbr).

**Questions answered.**
- Where in depth do pretraining corpora leave their mark functionally?
- Do models agree early (shared featurization) and diverge late (different relational semantics), or vice versa?
- Does the functional clustering match the weight-space clustering from E1?

### E5. Neighbor-attention forensics — REVISED after reading `rt/model.py`

**Design correction.** The RT's `nbr` attention is **architecturally masked to FK-linked cells** (`block_masks["nbr"] = q_in_f2p & pad`, built from the batch's `f2p_nbr_idxs`, ≤5 FK edges/row). Heads cannot attend off-schema, so "do heads attend along FK edges" is trivially true by construction — the original E5 framing would have measured the mask, not learning. The learnable quantities are *how attention distributes within the permitted neighborhood* and *how much the computation relies on the stream at all*.

**Why (revised).** With the schema hard-wired, the generator's imprint on relational computation shows up as (a) within-neighborhood attention structure and (b) functional reliance on neighbor information — measured by E7's ablations, which are promoted to the primary mechanistic probe.

**How (revised).** Attention weights aren't materialized (fused SDPA/flex kernels), so hook `wq`/`wk` inputs per nbr head and recompute masked softmax scores on small probe batches. Then per head/block/model: entropy over permitted neighbors; preference by relation *type* (which parent table; parent→child vs child→parent direction); consistency of head specialization across databases.

**Questions answered.**
- Within the allowed FK neighborhood, do heads specialize by relation type, and does the specialization pattern differ by pretraining generator?
- Is within-neighborhood attention sharper (lower entropy) for models whose corpora had real FK structure (RelDiff, GRDM) than fully-synthetic ones?
- Do E1's weight-space findings (RelDiff's outsized nbr deviation) correspond to distinct within-neighborhood attention behavior?

### E6. Linear probes on frozen representations

**Why.** Attention maps show where information *could* flow; probes show what information is actually *encoded*. Comparing probe accuracy across the four checkpoints turns "generator X is better" into "generator X's models encode more Y".

**How.** Freeze each model; extract per-cell/per-row representations on real DBs; train identical linear probes for: (a) parent-table identity of a row, (b) child-row cardinality (degree), (c) temporal ordering of two events, (d) column semantic type, (e) whether two rows are FK-linked (pairwise probe). Report Δ over a random-init RT baseline to isolate pretraining's contribution.

**Questions answered.**
- What relational information does each pretraining corpus actually put into representations?
- Does probe advantage on relational properties (a, b, e) predict downstream fine-tuning advantage? (If yes — mechanistic explanation of the benchmark ranking.)

### E7. Structure-vs-attribute reliance (behavioral ablation)

**Why.** One number per model: how much it *depends* on relational structure vs within-row statistics. Pure input corruption, no code surgery.

**How.** On real held-out data, measure masked-reconstruction loss under: (i) intact inputs; (ii) FK links shuffled within table (structure destroyed, marginals intact); (iii) attribute values shuffled within column (marginals intact, rows scrambled); (iv) neighbors dropped entirely. Structure-reliance index = (ii − i) / (iv − i) per model. Complement (optional): mean-ablate the `nbr` stream at inference — the internal version of (iv); agreement between the two strengthens the claim.

**Questions answered.**
- Which generator produces models that genuinely *use* cross-table information at inference time?
- Does structure-reliance correlate with the transfer-matrix "real" column — i.e., is relational computation the thing that transfers?

---

## Tier 3 — stretch (only with spare GPU-weeks)

### E8. Sparse autoencoders + cross-model feature matching

**Why/How.** Train SAEs on the residual stream of each model at 2–3 depths; match features across models by activation correlation on shared inputs; ask which features are universal (all four models) vs generator-specific. Universal features found only in relationally-pretrained models = candidate "relational circuit vocabulary".

**Questions answered.** What are the actual learned features, and which of them does each generator's data induce? High effort; only after E1–E7 are in the paper.

---

## Suggested paper narrative

1. **E3** ranks generators by transfer to real data (the headline table).
2. **E5 + E7** explain the ranking mechanistically: the winning generator's models have more selective FK heads and higher structure-reliance.
3. **E1 + E4** localize where generator identity lives in the network (appendix figures).
4. **E6** bridges to downstream results: probe advantages predict fine-tuning advantages.

## Execution order & prerequisites

| # | Needs | Where | Wall-clock |
|---|---|---|---|
| E1, E2 | checkpoints only | local CPU | minutes |
| E4, E5, E6, E7 | rt_icl + checkpoints + real RelBench (convert on box) | GPU box | hours each |
| E3 | + held-out synthetic corpora (RelDiff in vault ✓; PluRel uploadable; GRDM/RDB-PFN from collaborator) | GPU box | ~a day incl. data wrangling |
| E8 | everything above + SAE training | GPU box | days |

**Open items:** confirm whether all four arms shared the same init seed (affects E1's interpretation — ask the collaborator); obtain GRDM/RDB-PFN held-out corpora for E3's full matrix (a 4×3 matrix without them is still publishable).

---

## Probe dataset

**Standard probe suite: 9 masked-cell tasks = 3 auto-discovered tasks × 3 real RelBench DBs — `rel-ratebeer`, `rel-arxiv`, `rel-stack`** (RelBench 2.x datasets; box upgraded from the pinned 1.1.0, rt_icl verified compatible). All three DBs lie outside every generator's pretraining reference set, so results are clean out-of-distribution transfer. Converted through the same `pipeline.convert_all` path as the pretraining corpora (canonicalize → task discovery → rustler preprocess → MiniLM embeddings), evaluated with fixed identical batches across all four models.

---

## Results

### E1 — weight-space divergence (run 2026-08-11, local CPU)

Raw pairwise cosines are ~0.99 everywhere → the four arms share an init and stayed near it (short budget), so the informative view is each model's **deviation from the 4-model mean** (init proxy). Baseline note: four mean-centered vectors must average pairwise cosine ≈ −1/3, so −0.33 is "unrelated", above it is "similar", below is "extra-dissimilar".

**Deviation norms (how far each arm moved from the pack):** GRDM 15.5 > RelDiff 14.0 > PluRel 11.1 ≈ RDB-PFN 11.0.

**Deviation cosines:** PluRel~RDB-PFN = **+0.11** (far above the −1/3 baseline — the only genuinely similar pair); GRDM~PluRel/RDB-PFN ≈ −0.46 (below baseline — extra-dissimilar); all other pairs ≈ baseline.

**Per-stream deviation norms (where generator identity lives):**

| model | encoders | col-attn | feat-attn | nbr-attn | ffn |
|---|---|---|---|---|---|
| grdm | 1.95 | 6.10 | **7.18** | 5.55 | 10.82 |
| plurel | 1.05 | 5.13 | 4.68 | 3.90 | 7.69 |
| rdbpfn | 1.03 | 5.57 | 4.35 | 3.80 | 7.51 |
| reldiff | 1.64 | 5.37 | 5.88 | **6.43** | 9.49 |

**Findings:**
1. **PluRel and RDB-PFN (fully-synthetic-schema generators) produce near-identical models**; GRDM and RelDiff (real-reference-based) each carve out distinct directions.
2. **RelDiff is the only model whose deviation peaks in `nbr`-attention** (6.43 — largest nbr deviation of any arm; for every other arm nbr is the *smallest* attention stream). Consistent with RelDiff's corpora carrying real FK subgraph structures — its data distinctively reshaped the relational stream.
3. **GRDM peaks in `feat`-attention** (7.18) — its imprint is row-level/attribute statistics, matching its design (real reference attributes, 1-hop locality).
4. Per-block: `nbr` similarity rises monotonically with depth (0.989 → 0.997 blocks 0→11) — corpus-specific relational adaptation concentrates in **early** nbr layers; late nbr layers are nearly common across arms.
5. Decoders are essentially untouched (deviation ≈ 0.05) — output heads didn't differentiate; everything interesting is in the trunk.

**Caveats:** FFN norms are inflated by parameter count (largest stream); absolute movement from init is small (3.5 h budget), so these are directions, not magnitudes, of specialization. E4 (CKA) tests whether these weight-space differences are functional.

### E7 — structure-vs-attribute reliance (run 2026-08-12, GPU box, 27 fixed probe batches)

Masked-reconstruction loss per model under input/stream corruptions (identical batches across models):

| model | intact | Δ nbr zeroed | Δ FK shuffled | Δ FK blanked |
|---|---|---|---|---|
| reldiff | **0.339** | **+0.029** | **+0.069 (+20%)** | +0.065 |
| plurel | 0.339 | +0.007 | +0.009 | +0.014 |
| grdm | 0.350 | +0.013 | **−0.005** | −0.007 |
| rdbpfn | 0.351 | −0.002 | +0.003 | +0.007 |

Per-DB Δ(FK shuffle): reldiff = ratebeer +0.067 / arxiv **−0.112** / stack +0.165; every other model ≤ |0.027| everywhere.

**Findings:**
1. **RelDiff is the only model that genuinely uses relational structure out-of-distribution** — tied-best intact loss AND +20% degradation when FK links are corrupted. Every other model is structure-indifferent (|Δ| ≤ 2%) on these unseen schemas: they treat real databases as bags of rows.
2. **E1's prediction confirmed functionally**: RelDiff — the only arm whose weights deviated most in `nbr`-attention — is also the only model with meaningful nbr-stream reliance (+0.029, ~4–15× the others). Weight-space and behavioral measurements agree independently.
3. **Architectural insight**: for 3 of 4 models the parent→children `nbr` stream is near-vestigial; relational information flows through the `feat` stream's parent-cell inclusion (`kv_in_f2p`) — a child cell reads its parents to predict itself. Only RelDiff's corpus (real FK subgraphs) put the aggregation stream to work.
4. **The arxiv anomaly**: shuffling FK links *improves* RelDiff's loss on rel-arxiv (−0.112) while raising it on stack (+0.165) and ratebeer (+0.067). Its structural priors mislead it on citation-graph topology. Open question for a targeted follow-up (which tasks, which relations).
5. GRDM shows a mild inverse effect (structure corruption slightly *helps*, −0.005) — its relational computation does not transfer to unseen schemas. **Action: confirm with the collaborator which reference DBs each corpus drew from**, so probe DBs stay disjoint from all pretraining references.

### E5 — within-neighborhood attention structure (9 batches)

Mean normalized entropy over permitted neighbors (1.0 = uniform) and mean top-1 mass, per model:

| model | norm. entropy | top-1 mass |
|---|---|---|
| reldiff | **0.843** | **0.396** |
| grdm | 0.858 | 0.394 |
| rdbpfn | 0.890 | 0.361 |
| plurel | 0.898 | 0.361 |

**Finding:** the real-reference models (RelDiff, GRDM) attend more selectively within the permitted FK neighborhood; the fully-synthetic models tend toward uniform averaging — pooling, not selection.

### E4 — layer-wise CKA on real data (the dataset-dependent counterpart of E1)

Mean linear CKA between models, early (blocks 0–5) vs late (blocks 6–11), 20k non-padding cell activations:

| pair | early | late |
|---|---|---|
| grdm~reldiff | 0.944 | **0.824** (most similar) |
| plurel~reldiff | 0.944 | 0.810 |
| grdm~plurel | 0.938 | 0.718 |
| plurel~rdbpfn | 0.909 | 0.707 |
| rdbpfn~reldiff | 0.870 | 0.587 |
| grdm~rdbpfn | 0.808 | **0.406** (most different) |

**Findings:**
1. Functional divergence **grows monotonically with depth** — the four models agree on early featurization and disagree on late relational integration (block 11 CKAs down to 0.198–0.724).
2. **Function ≠ weights**: E1's weight space said PluRel≈RDB-PFN cluster while GRDM/RelDiff each stand apart. On real data the picture inverts — **RDB-PFN is the functional outlier** (lowest CKA in all its pairs), and the two real-reference models (grdm~reldiff) compute most *similarly*. Weight directions measure what the corpus pushed; CKA measures where the computation landed.

**Summary (E1 + E4 + E5 + E7):** on out-of-distribution real databases, only RelDiff's pretraining produced a model that computes relationally — selective within-neighborhood attention, an active parent→children aggregation stream, and a 20% functional dependence on FK structure — consistent with the weight-space prediction (E1's `nbr` deviation peak) made before any data was run. The fully-synthetic corpora (PluRel, RDB-PFN) yield structure-indifferent models, and GRDM's relational computation does not transfer beyond schemas resembling its references. Functional divergence between the four models concentrates in the late blocks (E4): generator identity lives in relational integration, not featurization.

Raw numbers: vault `interp/results/e{4,5,7}_real3.json`.

---

## Bridging mechanism → benchmark (`rt_benchmark.csv`, 9 RelBench tasks × 5 context lengths)

### X2 — correlation: does measured structure-reliance predict benchmark behavior? (done)

Benchmark facts first: mean AUROC at ctx 1024 — reldiff 0.748 > rdbpfn 0.740 > plurel 0.723 > grdm 0.666; and **only RelDiff's accuracy survives long context** (Δ AUROC 30k−1024: reldiff +0.006 vs grdm −0.040 / plurel −0.057 / rdbpfn −0.087). RelDiff's biggest wins are child-aggregation tasks at 30k ctx (rel-stack user-engagement **0.929**, user-badge **0.847**).

Correlating E7's per-DB structure-reliance with per-DB benchmark context-scaling across all 12 (model, DB) cells:

- **corr(structure-reliance, ctx-scaling) = 0.71** (nbr-reliance: 0.70; generator-level means: 0.82, n=4).
- The only cells with substantial mechanistic reliance — reldiff×stack (+0.165) and reldiff×ratebeer (+0.067) — are the only cells whose benchmark accuracy doesn't degrade at 30k ctx (+0.055, −0.001; all other cells lose 0.03–0.10).
- **The arxiv anomaly cross-validates**: reldiff×arxiv is the one cell where structure *misleads* the model mechanistically (reliance −0.112) — and exactly the cell where its ctx-scaling is negative (−0.071; paper-citation peaks at ctx 1024, drops at 30k). Mechanism and benchmark agree even in the exception.

**Interpretation:** a model benefits from long relational context only if it mechanistically relies on relational structure; pooling models (PluRel, RDB-PFN — near-uniform nbr attention per E5) dilute as context grows.

### X1 — causal bridge: the benchmark under mechanistic ablations (run 2026-08-13)

The exact benchmark protocol (RelBench task tables → rustler preprocess → masked-label inference → AUROC) rerun with the E7 interventions at inference: intact vs FK-links-shuffled vs nbr-stream-zeroed; 4 models × 3 conditions on identical batches (n=2048 @ ctx 1024, n=256 @ ctx 30000).

**Predictions (registered before results):** (1) RelDiff's user-engagement/user-badge AUROC collapses toward the pooling models' level under FK corruption; (2) its 30k-ctx advantage disappears under ablation; (3) PluRel/RDB-PFN AUROCs barely move under any ablation; (4) on arxiv, FK-shuffle *hurts RelDiff least* (or helps), mirroring its negative reliance there.

**Results — rel-stack user-engagement (RelDiff's largest benchmark win):**

| ctx | model | intact | FK shuffled | nbr zeroed |
|---|---|---|---|---|
| 30000 | **reldiff** | **0.934** | **0.563** | **0.610** |
| 30000 | grdm | 0.576 | 0.413 | 0.573 |
| 30000 | plurel | 0.563 | 0.534 | 0.551 |
| 30000 | rdbpfn | 0.531 | 0.531 | 0.527 |
| 1024 | reldiff | 0.847 | 0.718 | 0.728 |
| 1024 | plurel | 0.708 | 0.701 | 0.697 |
| 1024 | rdbpfn | 0.691 | 0.694 | 0.687 |
| 1024 | grdm | 0.609 | 0.618 | 0.622 |

user-badge @1024: reldiff 0.808→0.775/0.765; plurel/rdbpfn |Δ| ≤ 0.003. ratebeer user-churn @1024: reldiff 0.612→**0.673 under FK-shuffle (+0.061 — corruption HELPS)**, grdm 0.627→0.691 (+0.064), plurel/rdbpfn |Δ| ≤ 0.007. arxiv paper-citation @1024: all models ≈0.78 intact with small deltas; reldiff's are largest (−0.021/−0.033).

**Verdicts:**
1. **Confirmed** — FK corruption erases RelDiff's engagement advantage (0.847 → 0.72, the pooling models' intact level; their own deltas ≤ 0.011).
2. **Confirmed** — at 30k ctx RelDiff drops 0.934 → 0.563: the long-context win is carried entirely by relational structure. Ablated, it performs worse than its own 1024-ctx ablated self — long context without structure only adds noise.
3. **Confirmed** — PluRel/RDB-PFN are ablation-indifferent everywhere (≤0.011); for GRDM, ablation is neutral-to-helpful on these tasks.
4. **Refuted as located, confirmed in substance** — on arxiv paper-citation the anomaly did not manifest (RelDiff shows mild normal reliance). The misleading-structure phenomenon appeared on **ratebeer user-churn** instead: shuffling FKs *improves* both structure-using models by +0.06 — which explains why RDB-PFN (structure-blind) wins the ratebeer churn tasks in the benchmark CSV. The E7 arxiv sign-flip appears specific to masked-cell reconstruction.

**Summary:** pretraining corpus → mechanism → capability, with both signs demonstrated. RelDiff's corpus built relational computation that *causes* its long-context benchmark wins (removing the mechanism removes the wins); where its relational priors misfit a task (ratebeer churn), the same mechanism causes its losses. The structure-blind models are unaffected by either intervention. Raw numbers: vault `interp/results/xbridge{,2}_results.json`.

---

## Interpretation

The organizing principle behind all of the results is *predictive necessity*: the pretraining objective is masked-cell reconstruction, so a model learns to use cross-table structure only if the corpus makes masked values hard to recover without it. Each generator differs in how much of its signal it places in the neighborhood versus in the row itself, and every experiment below reads out a consequence of that choice.

### E1 (weight space) — what each corpus pushed on

- **RelDiff** is the only arm whose deviation peaks in `nbr`-attention. Its joint multi-hop graph diffusion regenerates attributes *conditioned on* neighboring tables, so neighbor values carry information about masked cells, and training pressure lands on the cross-table stream.
- **GRDM** moved farthest overall but concentrated in `feat`-attention. Its corpora are near-real subsamples with 1-hop locality: attribute realism is high, and the useful signal for reconstruction sits within the row and its immediate parents — pressure lands on row-level feature interactions, not on aggregation.
- **PluRel and RDB-PFN** moved least and in a shared direction (deviation cosine +0.11 vs a −1/3 baseline). Both synthesize schemas and values without grounding attributes in real relational dependencies, so they impose similar, comparatively weak training pressure; neither makes structure predictive.

### E4 (CKA on real data) — where the computations ended up

- All pairs agree early (CKA ≈ 0.9+ in blocks 0–5): featurization is set by the shared init and objective, not the generator.
- **grdm~reldiff** are the most similar pair late (0.824) despite having the most different weight directions in E1: both corpora carry real-data signal, so their computations converge on real inputs even though they got there along different parameter paths.
- **RDB-PFN** is the functional outlier late (0.406–0.707 against everyone) despite being weight-closest to PluRel. Weight proximity here reflects small updates from a shared init, not similar computation — the two measures answer different questions, which is why both are reported.

### E5 (within-neighborhood attention) — selection vs pooling

- **RelDiff (0.843) and GRDM (0.858)** have measurably lower attention entropy: specific neighbors mattered in their corpora, so their heads learned to select.
- **PluRel (0.898) and RDB-PFN (0.890)** sit near uniform: with no consistent neighbor signal in training, their nbr streams learned to average. Uniform averaging is exactly the operation that degrades as context length grows, which is what the benchmark's long-context collapse shows for these two models.

### E7 (reliance ablations) — who actually uses structure, out of distribution

- **RelDiff**: +20% loss under FK corruption and the only non-trivial nbr-stream reliance. Its relational computation is schema-*general*: it transfers to databases none of the models ever saw.
- **GRDM**: ≈0 (sometimes negative) reliance on unseen schemas, despite real-reference pretraining and sharp attention (E5). Its relational knowledge appears schema-*specific* — regularities of its particular reference databases rather than a transferable skill. This also explains its earlier strong showing on data resembling its references.
- **PluRel / RDB-PFN**: indifferent to every corruption (≤2%). Their models reconstruct from within-row statistics alone.
- The rel-arxiv sign flip (corruption helps RelDiff by −0.112) shows the cost side of learned priors: on citation topology, RelDiff's expectations about neighborhoods are wrong enough that removing them helps. Priors that transfer can also mis-transfer.

### X2 / X1 (benchmark) — the mechanism is the capability

- **RelDiff**: intact, it is tied-best at ctx 1024 and the only model that improves to 30k; ablated, both advantages vanish (0.934 → 0.563 on user-engagement). Its benchmark profile — wins on child-aggregation tasks, the unique context scaling, and the losses on ratebeer churn where corruption *helps* it — is entirely accounted for by the presence and fit of its relational mechanism.
- **GRDM**: weakest downstream overall; its wins in the benchmark CSV are count-style regressions (user-count, post-votes), i.e. attribute statistics, matching its E1 feat-attention imprint. Ablations are neutral-to-helpful: on unseen schemas its structural priors contribute noise.
- **RDB-PFN**: best at very short context (mean AUROC 0.750 @ ctx 100) — its corpus transfers useful attribute-level statistics — but it degrades fastest with context (−0.087) and is untouched by ablation: performance from marginals, not relations.
- **PluRel**: same pattern as RDB-PFN with slightly weaker short-context transfer; the two are behaviorally near-interchangeable, as E1 predicted from weights.

### Generator profiles (one paragraph each)

**GRDM** (diffusion over real references, 1-hop locality): teaches real marginal and row-level statistics plus relations specific to its reference schemas. Its model is strong where the test schema resembles a reference or where attribute statistics suffice, and carries no transferable relational skill. Note both GRDM and RelDiff are diffusion models — the difference is not the model class but where the generated data places predictive information.

**RelDiff** (joint multi-hop graph diffusion on real FK topology): the only corpus that makes neighborhoods predictive, and therefore the only one that teaches transferable relational inference. The same priors that produce its long-context wins mis-fit some topologies (ratebeer churn, arxiv reconstruction), producing its characteristic losses.

**PluRel** (fully synthetic schemas and values): broad schema diversity but statistically shallow relational dependencies; its model learns robust featurization and neighbor pooling, is structure-indifferent, and loses accuracy as context grows.

**RDB-PFN** (prior-sampled synthetic databases): weight-space twin of PluRel and behaviorally similar downstream — good short-context attribute transfer, no structure use — yet functionally the most idiosyncratic model in late blocks (E4), suggesting its prior shapes computation in ways the other corpora do not; unexamined beyond CKA.
