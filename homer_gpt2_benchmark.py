from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

SEED = 42
MODEL_NAME = "gpt2"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FULL_DIR = OUTPUT_ROOT / "full_finetuning"
LORA_DIR = OUTPUT_ROOT / "lora"

PROMPTS = [
    "### Human: Homer, what do you think about donuts?\n### Assistant:",
    "### Human: Homer, why don't you want to go to work?\n### Assistant:",
    "### Human: Homer, what is the meaning of life?\n### Assistant:",
]

GENERATION_KWARGS = {
    "max_new_tokens": 50,
    "temperature": 0.8,
    "top_p": 0.9,
    "do_sample": True,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_exp(x: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return math.exp(min(max(x, 0.0), 50.0))


def format_dialogue(text: str) -> str:
    text = (text or "").replace("\r\n", " ").replace("\n", " ")
    text = " ".join(text.split())
    if "### Human:" not in text or "### Assistant:" not in text:
        return f"### Human: {text} ### Assistant:"
    return text


@dataclass
class RunSummary:
    experiment: str
    total_parameters: int
    trainable_parameters: int
    trainable_percentage: float
    validation_loss: float
    validation_perplexity: float
    test_loss: float
    test_perplexity: float
    training_time_seconds: float
    saved_path: str
    peak_gpu_memory_mb: float | None = None
    model_size_mb: float | None = None


def get_param_counts(model: torch.nn.Module) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total) * 100.0 if total else 0.0
    return total, trainable, pct


def get_peak_gpu_memory_mb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def get_model_size_mb(path: Path) -> float:
    total = 0.0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 ** 2)


def load_dataset_split() -> DatasetDict:
    raw = load_dataset("Jim-Webfox/HomerSimpson")
    train_raw = raw["train"]
    cleaned = train_raw.map(lambda x: {"text": format_dialogue(x["text"])})

    train_val = cleaned.train_test_split(test_size=0.2, seed=SEED)
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=SEED)

    split = DatasetDict(
        {
            "train": train_val["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    out_meta = OUTPUT_ROOT / "dataset_split.json"
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_dataset": "Jim-Webfox/HomerSimpson",
                "seed": SEED,
                "train_rows": len(split["train"]),
                "validation_rows": len(split["validation"]),
                "test_rows": len(split["test"]),
                "max_length": MAX_LENGTH,
            },
            f,
            indent=2,
        )
    return split


def tokenize_split(dataset_split: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    def tokenize_batch(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        labels = []
        for row in enc["input_ids"]:
            labels.append([token if token != tokenizer.pad_token_id else -100 for token in row])
        enc["labels"] = labels
        return enc

    tokenized = DatasetDict(
        {
            "train": dataset_split["train"].map(tokenize_batch, batched=True, remove_columns=dataset_split["train"].column_names),
            "validation": dataset_split["validation"].map(tokenize_batch, batched=True, remove_columns=dataset_split["validation"].column_names),
            "test": dataset_split["test"].map(tokenize_batch, batched=True, remove_columns=dataset_split["test"].column_names),
        }
    )
    return tokenized


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_full_finetuning(tokenizer: AutoTokenizer, tokenized: DatasetDict) -> RunSummary:
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    total, trainable, pct = get_param_counts(model)

    output_dir = FULL_DIR / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    start = time.perf_counter()
    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    elapsed = time.perf_counter() - start

    final_model_dir = FULL_DIR / "final_model"
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    val_loss = float(val_metrics["eval_loss"])
    test_loss = float(test_metrics["eval_loss"])
    summary = RunSummary(
        experiment="full_finetuning",
        total_parameters=total,
        trainable_parameters=trainable,
        trainable_percentage=pct,
        validation_loss=val_loss,
        validation_perplexity=safe_exp(val_loss),
        test_loss=test_loss,
        test_perplexity=safe_exp(test_loss),
        training_time_seconds=elapsed,
        saved_path=str(final_model_dir),
        peak_gpu_memory_mb=get_peak_gpu_memory_mb(),
        model_size_mb=get_model_size_mb(final_model_dir),
    )
    save_json(FULL_DIR / "summary.json", summary)
    return summary


def run_lora(tokenizer: AutoTokenizer, tokenized: DatasetDict) -> RunSummary:
    device = get_device()
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.config.pad_token_id = tokenizer.eos_token_id
    base_model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(base_model, lora_config)

    total, trainable, pct = get_param_counts(model)

    output_dir = LORA_DIR / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    start = time.perf_counter()
    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    elapsed = time.perf_counter() - start

    adapter_dir = LORA_DIR / "adapters"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    val_loss = float(val_metrics["eval_loss"])
    test_loss = float(test_metrics["eval_loss"])
    summary = RunSummary(
        experiment="lora",
        total_parameters=total,
        trainable_parameters=trainable,
        trainable_percentage=pct,
        validation_loss=val_loss,
        validation_perplexity=safe_exp(val_loss),
        test_loss=test_loss,
        test_perplexity=safe_exp(test_loss),
        training_time_seconds=elapsed,
        saved_path=str(adapter_dir),
        peak_gpu_memory_mb=get_peak_gpu_memory_mb(),
        model_size_mb=get_model_size_mb(adapter_dir),
    )
    save_json(LORA_DIR / "summary.json", summary)
    return summary


def generate_responses(model, tokenizer, prompts: list[str]) -> None:
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(get_device())
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("-" * 80)
        print(text)
        print("-" * 80)


def save_json(path: Path, summary: RunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": summary.experiment,
        "total_parameters": summary.total_parameters,
        "trainable_parameters": summary.trainable_parameters,
        "trainable_percentage": summary.trainable_percentage,
        "validation_loss": summary.validation_loss,
        "validation_perplexity": summary.validation_perplexity,
        "test_loss": summary.test_loss,
        "test_perplexity": summary.test_perplexity,
        "training_time_seconds": summary.training_time_seconds,
        "saved_path": summary.saved_path,
        "peak_gpu_memory_mb": summary.peak_gpu_memory_mb,
        "model_size_mb": summary.model_size_mb,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_benchmark_table(full_summary: RunSummary, lora_summary: RunSummary) -> pd.DataFrame:
    records = [
        {
            "Experiment": "Full Fine-Tuning",
            "Total parameters": full_summary.total_parameters,
            "Trainable parameters": full_summary.trainable_parameters,
            "Trainable %": full_summary.trainable_percentage,
            "Validation loss": full_summary.validation_loss,
            "Validation perplexity": full_summary.validation_perplexity,
            "Test loss": full_summary.test_loss,
            "Test perplexity": full_summary.test_perplexity,
            "Training time (s)": full_summary.training_time_seconds,
            "Peak GPU memory (MB)": full_summary.peak_gpu_memory_mb,
            "Saved model size (MB)": full_summary.model_size_mb,
        },
        {
            "Experiment": "LoRA",
            "Total parameters": lora_summary.total_parameters,
            "Trainable parameters": lora_summary.trainable_parameters,
            "Trainable %": lora_summary.trainable_percentage,
            "Validation loss": lora_summary.validation_loss,
            "Validation perplexity": lora_summary.validation_perplexity,
            "Test loss": lora_summary.test_loss,
            "Test perplexity": lora_summary.test_perplexity,
            "Training time (s)": lora_summary.training_time_seconds,
            "Peak GPU memory (MB)": lora_summary.peak_gpu_memory_mb,
            "Saved model size (MB)": lora_summary.model_size_mb,
        },
    ]
    df = pd.DataFrame(records)
    out_csv = OUTPUT_ROOT / "benchmark_summary.csv"
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    tokenizer = build_tokenizer()
    split = load_dataset_split()
    tokenized = tokenize_split(split, tokenizer)

    print("Dataset structure:")
    print(tokenized)
    print("Sample validation row keys:", tokenized["validation"][0].keys())

    full_summary = run_full_finetuning(tokenizer, tokenized)
    lora_summary = run_lora(tokenizer, tokenized)

    print("\nFull Fine-Tuning summary:")
    print(json.dumps({
        "total_parameters": full_summary.total_parameters,
        "trainable_parameters": full_summary.trainable_parameters,
        "trainable_percentage": full_summary.trainable_percentage,
        "validation_loss": full_summary.validation_loss,
        "validation_perplexity": full_summary.validation_perplexity,
        "test_loss": full_summary.test_loss,
        "test_perplexity": full_summary.test_perplexity,
        "training_time_seconds": full_summary.training_time_seconds,
    }, indent=2))

    print("\nLoRA summary:")
    print(json.dumps({
        "total_parameters": lora_summary.total_parameters,
        "trainable_parameters": lora_summary.trainable_parameters,
        "trainable_percentage": lora_summary.trainable_percentage,
        "validation_loss": lora_summary.validation_loss,
        "validation_perplexity": lora_summary.validation_perplexity,
        "test_loss": lora_summary.test_loss,
        "test_perplexity": lora_summary.test_perplexity,
        "training_time_seconds": lora_summary.training_time_seconds,
    }, indent=2))

    benchmark_df = build_benchmark_table(full_summary, lora_summary)
    print("\nBenchmark table:")
    print(benchmark_df.to_string(index=False))

    print("\nGeneration with Full Fine-Tuning:")
    full_model = AutoModelForCausalLM.from_pretrained(FULL_DIR / "final_model").to(device)
    full_model.config.pad_token_id = tokenizer.eos_token_id
    generate_responses(full_model, tokenizer, PROMPTS)

    print("\nGeneration with LoRA:")
    base_lora_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    base_lora_model.config.pad_token_id = tokenizer.eos_token_id
    lora_model = PeftModel.from_pretrained(base_lora_model, str(LORA_DIR / "adapters")).to(device)
    generate_responses(lora_model, tokenizer, PROMPTS)


if __name__ == "__main__":
    main()
