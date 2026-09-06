# Next interpretability experiments

Written after reviewing the ATTRIB draft against `rt_benchmark.csv`. The paper argues a chain: the pretraining corpus builds a mechanism, and the mechanism produces the benchmark result. Two links in that chain are weaker than the draft admits, and both are fixable. The rest of this document lists what to run, in the order I would run it.

## Where things stand

Four RT checkpoints (12 blocks, d=256, 19.2M parameters) share architecture, step budget, and objective. Only the generator behind the pretraining corpus differs. Four measurements agree that RelDiff's corpus is the only one that taught the model to use foreign-key structure: its weights deviate most in `nbr`-attention (6.43 against 3.80 to 5.55), its within-neighborhood attention is sharpest (entropy 0.843 against 0.898 for PluRel), its reconstruction loss rises 20% when FK links are shuffled while the others move under 2%, and ablating that mechanism removes its benchmark advantage.

Two things are missing. The corpus end of the chain was never measured, only described from each generator's design. And the benchmark end rests, at 30k context, on very few positive examples.

| # | Experiment | Cost | Needs |
|---|---|---|---|
| 1 | Re-power the two long-context tasks | 2 to 3 GPU-hours | GPU box |
| 2 | Predictive necessity in the corpora | Hours, CPU | `data/*.zip` |
| 3 | Random-init floor | Under an hour | Checkpoints |
| 4 | Dose-response on FK corruption | ~1 GPU-hour | rt_icl |
| 5 | Separate `nbr` from `feat`'s parent path | ~1 GPU-hour | rt_icl |
| 6 | Cross-model stream transplant | Half a day | rt_icl |
| 7 | Linear probes (E6) | Half a day | rt_icl |
| 8 | Head knockout sweep | Half a day | rt_icl |
| 9 | The fifth arm | 3.5 h pretraining plus corpus prep | GPU box |
| 10 | Predict the anomaly from topology | Hours, CPU | Corpora, probe DBs |

## Before the deadline

### 1. Re-power the two long-context tasks

At 30k context the benchmark used n=256. On `rel-stack` user-engagement the positive rate is 0.023, so that AUROC of 0.929 rests on six positive examples. User-badge has ten. A Hanley-McNeil standard error puts the 0.929 at plus or minus 0.073, a 95% interval of [0.787, 1.07].

Those two tasks carry the paper's claim that RelDiff is the only model whose accuracy survives long context. Drop them and the picture changes:

| Change in mean AUROC, 1024 to 30k | reldiff | rdbpfn | plurel | grdm |
|---|---|---|---|---|
| All 6 classification tasks | +0.006 | -0.087 | -0.057 | -0.040 |
| 4 well-powered tasks | -0.019 | -0.084 | -0.041 | -0.044 |
| The 2 rare-label tasks | +0.056 | -0.092 | -0.089 | -0.031 |

RelDiff still degrades least on the well-powered subset, so the ordering holds and the argument survives. What does not survive is the absolute claim that RelDiff improves with context. That claim comes entirely from six and ten positives.

Rerun both tasks at 30k with n=2048 instead of 256, across all four models and all three ablation conditions. The logged runs took 30 to 60 seconds at n=256, so eight times the samples is roughly 5 to 8 minutes per cell, and 24 cells is 2 to 3 GPU-hours. This is the best use of remaining GPU time. It turns the weakest claim in the paper into the strongest one, or it tells you to reframe before a reviewer does.

If the GPU is unavailable, report confidence intervals and soften the claim to "degrades least."

### 2. Predictive necessity in the corpora

The draft explains everything through predictive necessity: a generator teaches structure use only when its corpus makes masked cells hard to recover without neighbors. That property is never measured. It is inferred from how each generator is built.

Measure it directly, with no RT involved. For each corpus, hold out cells and fit a gradient-boosted predictor twice, first on within-row features alone, then on within-row features plus parent and child aggregates. The gain from adding neighbors is the neighbor predictive information in that corpus. Report it per generator, and split by column type so numeric and categorical can differ.

The prediction is that RelDiff shows a large gain, GRDM a moderate one, PluRel and RDB-PFN close to zero. If that holds, the causal chain has a measured quantity at every node instead of a design description at the first one, and the attribution stops being about which generator and starts being about which property. For a workshop on data attribution that difference matters. Runs on CPU against the zips already in `data/`.

A negative result is also useful. If PluRel's corpus does carry neighbor information that its model failed to learn, the story changes from "the corpus lacked the signal" to "the signal was there and training missed it," which points at optimization rather than data.

### 3. Random-init floor

The attention entropies are 0.843, 0.858, 0.890, and 0.898. Nothing in the paper says what an untrained model scores, so a reader cannot tell whether PluRel at 0.898 learned a mild preference or learned nothing at all. If a random-init RT sits near 0.90, then three of the four models never moved their `nbr` stream and only RelDiff did, which is a much sharper statement than the current 0.055 gap.

Instantiate the RT with random weights and run the same E5, E4, and E7 probes. This also replaces the hack in E1, where the four-model mean stands in for the initialization, and it gives E4 the CKA floor the original plan called for but the results never reported. Under an hour, and it strengthens three tables at once.

### 4. Dose-response on FK corruption

E7 and X1 both compare intact against fully corrupted. Two points cannot show a trend. Corrupt 0, 25, 50, 75, and 100 percent of FK edges and plot loss against corruption fraction.

A monotone curve for RelDiff and flat lines for the other three is far harder to dismiss than a pair of numbers, because noise and batch effects do not usually produce monotone dose-response. Three extra inference passes per model.

### 5. Separate `nbr` from `feat`'s parent path

There is a contradiction between the notes and the paper. E7's third finding says that for three of the four models, relational information flows through the `feat` stream's parent-cell inclusion (`kv_in_f2p`) rather than through `nbr`. The paper says those models treat databases as bags of rows. Both cannot be true unless the FK shuffle also rewires `f2p` for the feat path.

Resolve it by ablating the two paths independently: zero `nbr` alone, block `kv_in_f2p` alone, then both. If PluRel's loss jumps when the parent path is blocked, "structure-indifferent" has to become "does not use the aggregation stream," which is a narrower claim than the one the abstract makes. Better to find this now than in review. Roughly a GPU-hour, since it is a masking change to code that already exists.

## If a GPU box frees up

### 6. Cross-model stream transplant

The four models have identical shapes, so activations move between them without any adapter. Run a batch through RelDiff, capture its `nbr` stream output, and patch it into PluRel's forward pass on the same batch. Then reverse it.

If PluRel's reconstruction loss or AUROC improves under RelDiff's neighbor activations, the relational computation is a transplantable component, and the corpus's contribution is localized to a specific set of activations rather than diffused through the network. If RelDiff degrades toward PluRel's level under the reverse patch, that is the same claim from the other side. Either result localizes the attribution far more precisely than CKA or entropy, both of which only measure similarity.

This is the experiment most likely to produce a result the ATTRIB audience has not seen before, since it treats a pretraining corpus as something that installs a swappable module.

### 7. Linear probes

Already scoped as E6 and never run. Freeze each model, extract representations on the three probe databases, and train identical linear probes for parent-table identity, child-row cardinality, temporal ordering, column type, and whether two rows are FK-linked. Report the gain over a random-init baseline so the probe measures pretraining rather than architecture.

Probes answer a different question than the ablations do. Ablation asks what the model uses; probing asks what it encodes. A model can encode parent identity and ignore it. If PluRel encodes relational facts it never uses, that separates "the corpus did not teach it" from "the objective did not require it."

### 8. Head knockout sweep

Knock out `nbr` heads one at a time in RelDiff and rank them by loss increase. Two outcomes are informative. If a handful of heads account for most of the reliance, there is a circuit to describe, and the paper can name it. If the effect is spread evenly across all heads, the mechanism is distributed, which is worth saying plainly rather than leaving implied.

Then check whether the same heads matter on all three probe databases. Heads that transfer across schemas are the strongest available evidence that RelDiff taught a general skill rather than a set of schema-specific regularities.

## After the deadline

### 9. The fifth arm

This is the experiment that would make it a full paper. Everything so far intervenes on the model. Nothing intervenes on the data, which is what a data attribution claim ultimately rests on.

Take the RelDiff corpus and shuffle its FK edges at generation time, so every marginal distribution is bit-identical and only the coupling between values and structure is destroyed. Pretrain a fifth RT with the same budget and objective. If the mechanism does not form, the property identified in experiment 2 is the cause, not a correlate. That is leave-one-property-out attribution applied to a corpus.

The reverse is stronger still and harder: add cross-table dependency to PluRel's value stage and show the mechanism appears. That requires generator surgery rather than a shuffle, but it would demonstrate the property is sufficient and not only necessary.

Pretraining is 3.5 hours. Preparing the corpus is the real cost.

### 10. Predict the anomaly from topology

Two sign flips are currently loose ends. FK shuffling improves RelDiff on `rel-arxiv` reconstruction by 0.112, and improves both structure-using models on ratebeer user-churn by about 0.06. The paper reports them honestly and explains neither.

The hypothesis worth testing is that RelDiff's priors assume a tree-like FK topology with informative parents, and that citation graphs violate it. Compute topological statistics for each probe database and for RelDiff's reference corpus, including degree distribution, fan-in, depth, and cyclicity, then test whether distance from the reference topology predicts the sign of reliance. If it does, the exceptions become predictions, and the paper gains a rule for when the mechanism will misfire instead of an apology for two cases where it did.

### 11. Everything else

The cross-generator transfer matrix (E3) was the original headline and still needs held-out corpora from the collaborators for GRDM and RDB-PFN. Per-relation-type reliance would connect to the temporal-archetype work already in `RFMs.md`, asking whether RelDiff relies on one-to-many links more than many-to-many, and on temporal relations more than atemporal ones. A logit lens over depth would show where each model resolves a masked cell and whether relational cells resolve later than marginal ones, which would give E4's late-block divergence a mechanism. Scaling the `nbr` stream by a gain above 1 in the pooling models would separate "the information is absent" from "the information is present and downweighted." Sparse autoencoders remain the stretch goal from the original plan.

## Paper fixes that need no new runs

Several problems in the draft are edits rather than experiments.

There are no inline citations. References [1] through [9] sit in the list and nothing in the body cites them.

The shared initialization is inferred from pairwise cosines near 0.99, not confirmed. `MECH_INTERP_EXPERIMENTS.md` still lists it as a question for the collaborator, and E1's whole interpretation depends on the answer. Either confirm it or state that it is inferred.

Table 4's intact values differ from `rt_benchmark.csv` (0.934 against 0.929, 0.847 against 0.852) because X1 is a separate rerun. Say so in a footnote, or a reviewer will read it as an inconsistency.

"Mean AUROC" covers the 6 classification tasks, not the 9 tasks named in the setup.

The diversity metric in Section 5 is a Euclidean distance in RFM embedding space, and the paper never says which model produced the embeddings. If it was one of the four checkpoints, RelDiff's top score may be an artifact of its own representation.

The corpus size per arm, 1k databases, appears nowhere in the paper.

Table 3 shows a "FK blanked" column that the text never discusses.

The paper has no figures. E1 is recomputable from `model_checkpoints/` on a laptop in minutes, so a stream-by-depth heatmap and a CKA-by-depth curve are both available cheaply.
