"""Utilities for LoRA fine-tuning.

This module provides a convenience function to fine-tune a causal language
model using Low-Rank Adaptation (LoRA).

Expected dataset format
------------------------
The ``dataset_dir`` argument should point to a directory containing one or
more JSON Lines (``.jsonl``) files. Each line must be a JSON object with a
``text`` field representing the training sample, for example::

    {"text": "Sample sequence one"}
    {"text": "Another training example"}

These files are loaded with :func:`datasets.load_dataset` using the ``json``
loader.
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def train_lora(
    model_path: str,
    dataset_dir: str,
    output_path: str,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 5e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> None:
    """Fine-tune a pretrained model with LoRA.

    Parameters
    ----------
    model_path:
        Path or model name of a pretrained causal language model.
    dataset_dir:
        Directory containing training data. See module docstring for the
        expected dataset format.
    output_path:
        Directory where the resulting LoRA adapter weights will be saved.
    batch_size, epochs, lr, lora_r, lora_alpha, lora_dropout:
        Training hyperparameters controlling batch size, number of epochs,
        learning rate and LoRA configuration.
    """

    data_files = {"train": os.path.join(dataset_dir, "*.jsonl")}
    dataset = load_dataset("json", data_files=data_files)["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize(batch: dict[str, str]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, max_length=tokenizer.model_max_length)

    tokenized = dataset.map(_tokenize, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(model_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    parser.add_argument("model_path", help="Path or identifier of the base model")
    parser.add_argument("dataset_dir", help="Directory containing .jsonl training files")
    parser.add_argument("output_path", help="Directory to save the LoRA adapter")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout probability")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_lora(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
