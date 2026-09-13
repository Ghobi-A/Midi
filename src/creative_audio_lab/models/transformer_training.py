"""Reproducible training, evaluation, and safe artefacts for the neural baseline."""

from __future__ import annotations

import copy
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .neural_events import EOS, PAD, NeuralNoteEvent, NoteEventCodec
from .transformer_melody import TransformerConfig, TransformerMelodyModel

ARTEFACT_FORMAT = "creative-audio-lab.transformer-melody/1"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    patience: int = 4
    seed: int = 0
    device: str = "auto"


class MelodyWindowDataset(Dataset):
    """Windows made independently inside compositions (never across split/piece boundaries)."""

    def __init__(self, sequences: Sequence[Sequence[NeuralNoteEvent]], context_length: int) -> None:
        self.samples: List[Tuple[List[NeuralNoteEvent], List[NeuralNoteEvent]]] = []
        for sequence in sequences:
            for start in range(0, max(len(sequence) - 1, 0), context_length):
                block = list(sequence[start:start + context_length + 1])
                if len(block) >= 2:
                    self.samples.append((block[:-1], block[1:]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _collate(batch):
    width = max(len(inputs) for inputs, _ in batch)
    pad = NeuralNoteEvent(PAD, PAD, PAD)
    inputs, targets = [], []
    for source, target in batch:
        inputs.append(source + [pad] * (width - len(source)))
        targets.append(target + [pad] * (width - len(target)))
    def tensor(field, rows):
        return torch.tensor([[getattr(event, field) for event in row] for row in rows], dtype=torch.long)
    return ({field: tensor(field, inputs) for field in ("pitch", "duration", "velocity")},
            {field: tensor(field, targets) for field in ("pitch", "duration", "velocity")})


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(sequences, context_length, batch_size, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(MelodyWindowDataset(sequences, context_length), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=_collate, generator=generator)


def evaluate_transformer(model: TransformerMelodyModel,
                         sequences: Sequence[Sequence[NeuralNoteEvent]],
                         *, batch_size: int = 32, device: str = "cpu") -> Dict[str, float]:
    """Score true note attributes in the same summed bits/note unit as the n-gram."""
    model.eval()
    target_device = _device(device)
    model.to(target_device)
    sums = {name: 0.0 for name in ("pitch", "duration", "velocity")}
    notes = 0
    with torch.no_grad():
        for source, target in _loader(sequences, model.config.context_length, batch_size):
            source = {key: value.to(target_device) for key, value in source.items()}
            target = {key: value.to(target_device) for key, value in target.items()}
            logits = model(**source, padding_mask=source["pitch"].eq(PAD))
            note_mask = target["pitch"].ne(PAD) & target["pitch"].ne(EOS)
            count = int(note_mask.sum())
            notes += count
            for name, output in zip(("pitch", "duration", "velocity"), logits):
                loss = F.cross_entropy(output.transpose(1, 2), target[name], reduction="none")
                sums[name] += float(loss[note_mask].sum())
    if not notes:
        return {"notes": 0, "bits_per_note": float("nan"),
                **{f"{name}_bits_per_note": float("nan") for name in sums},
                "perplexity_per_note": float("nan")}
    components = {f"{name}_bits_per_note": value / math.log(2) / notes
                  for name, value in sums.items()}
    total = sum(components.values())
    return {"notes": notes, **components, "bits_per_note": total,
            "perplexity_per_note": 2.0 ** total}


def train_transformer(train_sequences: Sequence[Sequence[NeuralNoteEvent]],
                      val_sequences: Sequence[Sequence[NeuralNoteEvent]],
                      model_config: TransformerConfig, training: TrainingConfig):
    """Fit with AdamW and validation-only early stopping; return the frozen best model."""
    random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass
    device = _device(training.device)
    model = TransformerMelodyModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate,
                                  weight_decay=training.weight_decay)
    loader = _loader(train_sequences, model_config.context_length, training.batch_size,
                     shuffle=True, seed=training.seed)
    if not len(loader.dataset):
        raise ValueError("training split contains no next-note examples")
    history, best_state, best_epoch, best_bits, stale = [], None, 0, float("inf"), 0
    for epoch in range(1, training.epochs + 1):
        started, total_loss, batches = time.monotonic(), 0.0, 0
        model.train()
        for source, target in loader:
            source = {key: value.to(device) for key, value in source.items()}
            target = {key: value.to(device) for key, value in target.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**source, padding_mask=source["pitch"].eq(PAD))
            loss = sum(F.cross_entropy(output.transpose(1, 2), target[name], ignore_index=PAD)
                       for name, output in zip(("pitch", "duration", "velocity"), logits))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        validation = evaluate_transformer(model, val_sequences,
                                          batch_size=training.batch_size, device=str(device))
        row = {"epoch": epoch, "training_loss": total_loss / batches,
               "validation_loss": validation["bits_per_note"] * math.log(2),
               **{f"validation_{key}": value for key, value in validation.items()},
               "learning_rate": optimizer.param_groups[0]["lr"],
               "runtime_seconds": time.monotonic() - started,
               "training_notes": sum(max(len(s) - 2, 0) for s in train_sequences),
               "training_sequences": len(train_sequences)}
        history.append(row)
        if validation["bits_per_note"] < best_bits:
            best_bits, best_epoch = validation["bits_per_note"], epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= training.patience:
                break
    if best_state is None:
        raise ValueError("validation split contains no scoreable notes")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_transformer_artefact(path: str | Path, model: TransformerMelodyModel,
                              codec: NoteEventCodec, metadata: Dict[str, Any]) -> Path:
    """Write JSON metadata plus a tensor-only state_dict (never a pickled model object)."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"format": ARTEFACT_FORMAT, "architecture": model.config.to_dict(),
               "note_event_codec": codec.to_dict(), "pytorch_version": torch.__version__,
               "python_version": platform.python_version(), "git_commit": git_commit(),
               "created_at": datetime.now(timezone.utc).isoformat(), **metadata}
    (directory / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                         allow_nan=False) + "\n")
    torch.save(model.state_dict(), directory / "weights.pt")
    return directory


def load_transformer_artefact(path: str | Path, map_location: str = "cpu"):
    directory = Path(path)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("format") != ARTEFACT_FORMAT:
        raise ValueError("Unsupported Transformer artefact format")
    model = TransformerMelodyModel(TransformerConfig(**metadata["architecture"]))
    try:
        state = torch.load(directory / "weights.pt", map_location=map_location, weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "Safe neural artefact loading requires a PyTorch release with weights_only support"
        ) from error
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("weights file is not a state_dict")
    model.load_state_dict(state)
    model.eval()
    return model, NoteEventCodec.from_dict(metadata["note_event_codec"]), metadata
