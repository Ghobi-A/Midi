# Real MIDI corpus quickstart

The repository can now bootstrap real MIDI corpora into a local, gitignored cache and write the manifest expected by the existing experiment pipeline.

## Recommended first run

```bash
python scripts/fetch_midi_corpora.py maestro commu
```

This downloads and extracts:

- **MAESTRO v3 MIDI-only** from the canonical Magenta archive. The archive SHA-256 is verified before extraction.
- **ComMU** raw MIDI plus `commu_meta.csv` from the official ComMU repository.

The resulting files live under:

```text
data/corpora/
├── maestro/
├── commu/
└── manifest.json
```

`data/corpora/` is intentionally gitignored. The raw binary corpus does not need to be committed for experiments to be reproducible: model artefacts already record corpus hashes and split provenance.

## Run the existing real-data experiment

```bash
python scripts/run_real_data_experiment.py \
  --manifest data/corpora/manifest.json \
  --output-dir experiments/real-midi
```

That uses the existing composition-level train/validation/test split, selects the statistical baseline on validation data, evaluates the frozen winner on test, and writes the experiment artefacts already defined by the project.

## Add GiantMIDI-Piano

GiantMIDI-Piano's official distribution is a Google Drive folder. Install the small optional download helper and fetch all supported corpora:

```bash
pip install -e ".[data]"
python scripts/fetch_midi_corpora.py all
```

Or fetch only GiantMIDI:

```bash
python scripts/fetch_midi_corpora.py giantmidi
```

The bootstrap accepts either the full GiantMIDI corpus or its curated subset and checks that at least the curated-scale number of MIDI files is present after download/extraction.

## Re-fetch or retain archives

Use `--force` to delete and rebuild selected extracted corpus directories. Download archives are deleted after successful extraction by default; add `--keep-archives` to retain them.

```bash
python scripts/fetch_midi_corpora.py maestro commu --force --keep-archives
```

## Scope

This bootstrap deliberately does not change the modelling architecture or relax the existing dataset/provenance checks. Its job is only to turn canonical public dataset distributions into the local directory + manifest contract the project already knows how to ingest.
