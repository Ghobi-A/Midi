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
