"""Utilities for audio upscaling models.

This module centralises helper routines shared by the different
upscaling modules:

* :func:`load_model` attempts to load a pre-trained neural network using
  either the `ddsp` or `rave` libraries.  The function returns a simple
  callable that maps an ``(audio, sample_rate)`` pair to an enhanced
  audio array.  If the third party libraries or checkpoints are not
  available in the execution environment a no-op model is returned so
  that callers can still function without hard failures.
* :func:`resample` performs sample-rate conversion using ``librosa`` when
  available and falls back to ``scipy``.

The implementations intentionally avoid importing heavy dependencies at
module import time; they are only pulled in when the helpers are
invoked.
"""
from __future__ import annotations

from typing import Callable
import logging
import numpy as np


def load_model(name: str) -> Callable[[np.ndarray, int], np.ndarray]:
    """Return a callable representing a pre-trained upscaling model.

    Parameters
    ----------
    name:
        Identifier of the model to load.  Currently ``"ddsp"`` and
        ``"rave"`` are understood.

    Returns
    -------
    Callable[[np.ndarray, int], np.ndarray]
        A function that takes an audio array and its sample rate and
        returns an enhanced array.  When the requested backend is not
        available an identity function is returned instead of raising an
        exception.  This keeps the rest of the codebase lightweight while
        still allowing optional high quality models when the environment
        provides them.
    """

    backend = name.lower()

    if backend == "ddsp":
        try:  # pragma: no cover - optional dependency
            import tensorflow as tf  # type: ignore
            from ddsp.colab import colab_utils  # type: ignore

            model, _ = colab_utils.load_pretrained()

            def apply(audio: np.ndarray, sr: int) -> np.ndarray:
                audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
                outputs = model(audio_tensor[None, :], training=False)
                audio_out = outputs["audio"] if isinstance(outputs, dict) else outputs
                return np.array(audio_out[0])

            return apply
        except Exception as exc:  # pragma: no cover - library not installed
            logging.warning("DDSP model unavailable: %s", exc)

    elif backend == "rave":
        try:  # pragma: no cover - optional dependency
            import torch  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
            from rave import RAVE  # type: ignore

            ckpt = hf_hub_download("acids-ircam/RAVE-v2", "rave-v2.ckpt")
            model = RAVE.load_from_checkpoint(ckpt)
            model.eval()

            def apply(audio: np.ndarray, sr: int) -> np.ndarray:
                with torch.no_grad():
                    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                    out = model(tensor)[0].cpu().numpy()
                return out

            return apply
        except Exception as exc:  # pragma: no cover - library not installed
            logging.warning("RAVE model unavailable: %s", exc)
    else:
        raise ValueError(f"Unsupported model '{name}'")

    # Fallback identity model
    return lambda audio, sr: audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample ``audio`` from ``orig_sr`` to ``target_sr``.

    ``librosa`` is used when available; otherwise a ``scipy`` based
    implementation is attempted.  If neither backend is installed the
    function raises a ``RuntimeError``.
    """

    if orig_sr == target_sr:
        return audio

    try:  # pragma: no cover - optional dependency
        import librosa  # type: ignore

        return librosa.resample(audio, orig_sr, target_sr)
    except Exception:  # pragma: no cover - library not installed
        try:
            from scipy.signal import resample as scipy_resample  # type: ignore
        except Exception as exc:  # pragma: no cover - library not installed
            raise RuntimeError("No resampling backend available") from exc

        n_samples = int(round(len(audio) * target_sr / orig_sr))
        return scipy_resample(audio, n_samples)
