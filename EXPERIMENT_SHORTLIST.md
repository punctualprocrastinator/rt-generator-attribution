# Experiment shortlist

Grouped by when to run them. One paragraph each. The three worth prioritizing are the corpus measurement, the stream transplant, and the fifth arm, for reasons given below.

## Before the deadline

### Measure predictive necessity in the corpora

The organizing principle behind the paper is that a generator teaches structure use only when its corpus makes masked cells unrecoverable from the row alone. Nothing in the work measures this in the data. Fit LightGBM to predict held-out cells twice, first from within-row features, then from the row plus parent and child aggregates. The lift between the two is the neighbor predictive information, reported per generator. It runs on CPU against the zips in `data/`. Right now the corpus node in the causal chain is a description of how each generator was built, and this turns it into a measured quantity. For ATTRIB that is the difference between attributing a result to a generator's name and attributing it to a generator's property. Run this one first.

### Add a random-init floor

An entropy of 0.843 against 0.898 means nothing without knowing what an untrained model scores. If random init sits near 0.90, then PluRel and RDB-PFN never moved their `nbr` stream and only RelDiff did. Same numbers, sharper claim. It costs almost nothing, it replaces E1's use of the four-model mean as an initialization proxy, and it gives E4 the CKA floor the original plan asked for.

### Corrupt foreign keys by degrees rather than all at once

Shuffle 0, 25, 50, 75, and 100 percent of FK edges instead of comparing intact against fully corrupted. A monotone curve for RelDiff and flat lines for the other three is much harder to argue with than two points, because noise rarely produces a monotone dose-response. It costs a few extra inference passes.

### Ablate `nbr` and `feat`'s `kv_in_f2p` separately

E7's third finding says relational information reaches three of the four models through the `feat` stream's parent-cell inclusion rather than through `nbr`, while the paper calls those same models bags of rows. Ablating each path on its own settles which channel carries structure in each model, and whether the paper's claim has to narrow to "does not use the aggregation stream".

## If a GPU box is free

### Transplant the `nbr` stream between models

The four checkpoints have identical shapes, so activations move between them with no adapter. Patch RelDiff's `nbr`-stream output into PluRel's forward pass on the same batch, then run it in reverse. If PluRel improves, the relational computation is a localized module that the corpus installed. If RelDiff degrades toward PluRel under the reverse patch, that is the same finding from the other direction. CKA and entropy only measure similarity, whereas this localizes the corpus's contribution to a specific set of activations. Of the GPU experiments this is the one most likely to produce something the ATTRIB audience has not seen before.

### Linear probes (E6)

Scoped in the notes already and never run. Freeze each model and train identical linear probes for parent-table identity, child-row degree, and whether two rows are FK-linked, reporting the gain over a random-init baseline. This turns "generator X is better" into "generator X's models encode parent identity and degree", which is the form an attribution claim should take.

### Knock out `nbr` heads one at a time

Rank RelDiff's `nbr` heads by the loss increase their removal causes. If a few heads carry most of the reliance, there is a circuit to describe and name. If the effect spreads evenly across heads, the mechanism is distributed and the paper should say so rather than leave it implied. Then check whether the same heads matter across all three probe databases, since heads that transfer across schemas are the best evidence that RelDiff taught a general skill.

## Camera-ready or the follow-up paper

### The fifth arm

Every experiment so far intervenes on the model. This one intervenes on the data, which is what a data attribution claim rests on. Shuffle the FK edges in RelDiff's corpus at generation time so that every marginal stays bit-identical and only the coupling between values and structure is destroyed, pretrain an identical RT, and show that the mechanism never forms. That is leave-one-property-out attribution applied to a corpus instead of a model, and it is the experiment that would make this a full paper rather than a workshop paper. Pretraining takes 3.5 hours and preparing the corpus is the real work. Nothing else on this list is worth as much.

### Turn the anomaly into a prediction

Measure the topological distance between each probe database and RelDiff's reference corpus, using degree distribution, fan-in, and cyclicity, then test whether that distance predicts the sign of the reliance. If it does, the arxiv and ratebeer sign flips stop being exceptions and become confirmations.

### Smaller items

The E3 transfer matrix still needs held-out corpora from the collaborators for GRDM and RDB-PFN. Per-relation-type reliance would connect to the temporal-archetype work in `RFMs.md`, asking whether RelDiff leans on one-to-many links more than many-to-many. A logit lens over depth would show where in the network each model resolves a masked cell. Sparse autoencoders remain the stretch goal from the original plan.
