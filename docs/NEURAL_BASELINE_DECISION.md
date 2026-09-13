# Neural baseline promotion decision

**Decision: NOT PROMOTED (real-data experiment pending).**

## Promotion gate

| Criterion | Required evidence | Current status |
| --- | --- | --- |
| Rights-cleared real corpus processed | Manifest, corpus hash and intake report | **Fail:** no corpus ships with the repository |
| Leakage controls pass | Composition split plus duplicate/related-work review | **Partial:** exact split isolation is tested; real-corpus semantic deduplication is pending |
| Reproducible train/validation/test metrics | Immutable run JSON and selected artefact | **Fail:** synthetic fixtures are tests, not results |
| Statistical baseline established | Orders 1–3 selected on validation and scored once on test | **Mechanism passes; real run pending** |
| Experiment tracking works | IDs, commit, corpus/split/config/seeds/runtime/paths | **Pass in automated machinery** |
| Generation evaluation and errors work | Per-sample records, mean/spread/n and interpretable flags | **Pass in machinery; perceptual calibration pending** |
| Corpus large enough | At least thousands of useful note events and enough independent compositions for stable held-out estimates | **Unknown** |

No Transformer should be implemented while the first, third and final gates
are unresolved. Doing so would optimize model complexity without knowing the
data regime or baseline.

## Smallest credible model after promotion

If every gate passes, the first neural comparison should be a decoder-only
Transformer over the existing symbolic token vocabulary: 4 layers, model
width 256, 8 attention heads, context 512, dropout 0.1, and fewer than roughly
10 million parameters. Train with next-token cross entropy, fixed seeds,
early stopping on validation loss, and the identical composition split. Tune
only learning rate and dropout on validation. Open test exactly once after the
configuration is frozen. Compare pitch cross entropy/perplexity/OOV to the
selected n-gram, then run the same fixed generation suite and error analysis.
This is a design, not an implementation or claim that the data supports it.

## 2026-09-12 implementation update

The implementation gate has now been opened without changing the evidence gate: the
repository contains a deliberately small causal Transformer baseline, but no
real-corpus score is claimed. It predicts the pitch, quantised duration, and velocity
bin of each next note from one shared representation of complete preceding notes.
This is the same three-factor note likelihood reported by the factorised n-gram, so
both sum the three negative log2 probabilities and divide by true held-out notes.

Architecture defaults are four pre-normalised Transformer layers, width 192, six
heads, feed-forward width 768, dropout 0.1, and 128-note context. MIDI pitch and the
tokenizer's bounded duration/velocity domains define fixed vocabularies; validation
and test frequencies never define IDs. PAD/BOS/EOS are explicit. AdamW training,
gradient clipping, fixed seeds, padding/causal masks and validation-only early
stopping are implemented in `models/transformer_training.py`.

A neural artefact is a directory containing JSON metadata and a tensor-only
`state_dict`. Loading reconstructs the recorded architecture and requests PyTorch
`weights_only=True`; it never deserialises a model object. Metadata records corpus
and licences, every composition split, seeds, software/git versions, hyperparameters,
history, selected epoch, validation metrics, and the once-only final test result.

This is **implemented infrastructure plus synthetic CI smoke evidence**, not a
published benchmark. Promotion remains pending a complete, rights-cleared corpus run
and duplicate-family review. Full-arrangement, text-conditioned, learned harmony,
bass/drums, large-scale training, hosting, and preference optimisation remain deferred.
