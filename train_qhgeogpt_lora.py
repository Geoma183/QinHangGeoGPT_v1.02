"""
train_qhgeogpt_lora.py

Instruction-tuning script for QHGeoGPT using LoRA adapters on top of a
4-bit quantized base model (e.g., DeepSeek-R1-7B variants).

This script:
- Loads an instruction-tuning dataset from a JSONL file.
- Filters overly long samples.
- Splits into train/eval sets.
- Loads a 4-bit base model and an existing LoRA adapter.
- Performs further instruction fine-tuning.
- Saves the updated LoRA adapter and tokenizer.

Author: (QHGeoGPT team)
"""

import os
import sys
import json
import logging
import argparse

import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training


# ------------------ Logging ------------------ #

def setup_logger(log_file: str = "qhgeogpt_lora_training.log") -> logging.Logger:
    """
    Configure console + file logging in UTF-8, with a simple, English-only format.
    """
    # Ensure stdout can handle UTF-8 (mainly for Windows)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # Not critical if this fails (e.g. older Python)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


# ------------------ Argument Parsing ------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instruction-tuning QHGeoGPT with LoRA on a 4-bit base model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/training/final_unified_instruction_finetune.jsonl",
        help="Path to the JSONL file containing instruction-tuning data.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model name or local path (e.g., 'DeepSeek-R1-7B' or local dir).",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to an existing LoRA adapter to continue fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/qhgeogpt_lora_v1",
        help="Directory to save the fine-tuned LoRA adapter and tokenizer.",
    )
    parser.add_argument(
        "--max-sample-char-len",
        type=int,
        default=5000,
        help="Maximum (instruction + output) character length to keep a sample.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train/eval batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    return parser.parse_args()


# ------------------ Data Loading ------------------ #

def load_instruction_data(
    data_path: str,
    max_sample_char_len: int,
    logger: logging.Logger,
) -> tuple[Dataset, Dataset]:
    """
    Load JSONL instruction-tuning data and split into train/eval datasets.

    Expected JSONL schema:
    {
        "instruction": "...",
        "input": "...",      # may be empty string
        "output": "..."
    }
    """
    logger.info(f"Loading data from {data_path} ...")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    # Filter samples that are too long
    filtered = [
        item
        for item in raw_data
        if len(item.get("instruction", "")) + len(item.get("output", "")) < max_sample_char_len
    ]
    df = pd.DataFrame(filtered)
    logger.info(f"Number of valid samples after length filtering: {len(df)}")

    if len(df) == 0:
        raise ValueError("No valid samples found. Check your data and filtering threshold.")

    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    return train_dataset, eval_dataset


# ------------------ Tokenization ------------------ #

def build_tokenize_fn(tokenizer, max_seq_length: int):
    """
    Build a tokenization function that formats instruction/input/output into a single
    causal language modeling sequence, and uses input_ids as labels.
    """

    def tokenize_function(examples):
        formatted_texts = []
        for ins, inp, out in zip(
            examples.get("instruction", []),
            examples.get("input", []),
            examples.get("output", []),
        ):
            ins = ins or ""
            inp = inp or ""
            out = out or ""
            if inp:
                text = f"Instruction: {ins}\nInput: {inp}\nResponse: {out}"
            else:
                text = f"Instruction: {ins}\nResponse: {out}"
            formatted_texts.append(text)

        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        # For causal LM, labels are typically the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return tokenize_function


# ------------------ Main ------------------ #

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    logger = setup_logger()

    logger.info("Starting QHGeoGPT LoRA instruction-tuning pipeline.")
    logger.info(f"Data path   : {args.data_path}")
    logger.info(f"Base model  : {args.base_model}")
    logger.info(f"LoRA path   : {args.lora_path}")
    logger.info(f"Output dir  : {args.output_dir}")

    # 1. Load data
    train_dataset, eval_dataset = load_instruction_data(
        data_path=args.data_path,
        max_sample_char_len=args.max_sample_char_len,
        logger=logger,
    )

    # 2. Load tokenizer from LoRA or base model (prefer LoRA path)
    logger.info("Loading tokenizer ...")
    tokenizer_source = args.lora_path if os.path.isdir(args.lora_path) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Prepare 4-bit quantization config
    logger.info("Setting up 4-bit quantization config ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_use_nested_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 4. Load base model in 4-bit
    logger.info("Loading base model in 4-bit ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 5. Prepare model for k-bit training and enable gradient checkpointing
    base_model = prepare_model_for_kbit_training(base_model)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False  # required when using gradient checkpointing

    # 6. Load existing LoRA adapter
    logger.info("Loading existing LoRA adapter ...")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_path,
    )

    # 7. Tokenization / dataset mapping
    logger.info("Tokenizing datasets ...")
    tokenize_fn = build_tokenize_fn(tokenizer, args.max_seq_length)

    train_dataset_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["instruction", "input", "output"],
    )
    eval_dataset_tokenized = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["instruction", "input", "output"],
    )

    # Optionally limit eval size for faster dev runs
    if len(eval_dataset_tokenized) > 50:
        eval_dataset_tokenized = eval_dataset_tokenized.select(range(50))

    # 8. Training arguments
    logger.info("Configuring TrainingArguments ...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        logging_steps=20,
        warmup_steps=50,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # use fp16 if GPU is available
        report_to="none",
        load_best_model_at_end=False,
    )

    # 9. Trainer
    logger.info("Initializing Trainer ...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_dataset_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 10. Train
    logger.info("Starting LoRA instruction fine-tuning ...")
    trainer.train()

    # 11. Save LoRA adapter and tokenizer
    logger.info("Training finished. Saving adapter and tokenizer ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model and tokenizer saved to: {args.output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
