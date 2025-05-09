"""
Supervised Fine-Tuning (SFT) script for the DeepSeek R1 GRPO pipeline.

Loads a base model, applies LoRA (optional), loads a dataset, 
and performs SFT using Hugging Face Transformers Trainer or TRL SFTTrainer.
"""

import logging
import os
import json
import argparse
from typing import Dict, Optional
import sys

import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from trl import SFTTrainer
from sklearn.metrics import accuracy_score

# Import utilities from the src directory
from model_utils import (
    load_base_model_and_tokenizer,
    apply_lora,
    save_model_and_tokenizer,
    set_special_tokens
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Custom Metrics Logging Callback ---
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs['step'] = state.global_step
            self.records.append(logs)

    def on_train_end(self, args, state, control, **kwargs):
        df = pd.DataFrame(self.records)
        output_csv = os.path.join(args.output_dir, "sft_training_metrics.csv")
        df.to_csv(output_csv, index=False)
        logger.info(f"Training metrics saved to {output_csv}")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = logits.argmax(-1)
    # Ensure labels are numpy arrays for accuracy_score
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    return {"eval_accuracy": accuracy_score(labels.flatten(), predictions.flatten())}

def run_sft_training(config: Dict) -> None:
    logger.info("--- Starting SFT Training --- ")
    logger.info(f"Configuration: {config}")

    # --- 1. Load Model and Tokenizer ---
    model_name_or_path = config.get('model_name') or config.get('model_name_or_path')
    if not model_name_or_path:
        raise ValueError("You must provide 'model_name' or 'model_name_or_path' in the config.")
    use_4bit = config.get('use_4bit', False)
    bnb_config_args = config.get('bnb_config', {})

    logger.info(f"Loading model: {model_name_or_path}")
    model, tokenizer = load_base_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        use_4bit=use_4bit,
        **bnb_config_args
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # --- 2. Handle Special Tokens (Optional) ---
    special_tokens_dict = config.get('special_tokens')
    if special_tokens_dict:
        set_special_tokens(tokenizer, special_tokens_dict)
        logger.info("Resizing model token embeddings after adding special tokens.")
        model.resize_token_embeddings(len(tokenizer))
        if model.config.pad_token_id != tokenizer.pad_token_id:
            logger.warning(f"Model pad token ID mismatch. Setting model's pad_token_id to tokenizer's.")
            model.config.pad_token_id = tokenizer.pad_token_id

    # --- 3. Apply LoRA ---
    use_lora = config.get('use_lora', False)
    is_lora_applied = False
    if use_lora:
        lora_config_dict = config.get('lora_config')
        if lora_config_dict:
            model = apply_lora(model, lora_config_dict, use_4bit=use_4bit)
            is_lora_applied = True
        else:
            logger.warning("use_lora is True, but no lora_config provided. Skipping LoRA.")

    # --- 4. Load and Prepare Dataset ---
    dataset_name = config.get('dataset_name')
    if dataset_name:
        raw_dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Dataset loaded from Hugging Face: {dataset_name}")
    else:
        dataset_path = config.get('dataset_path')
        if not dataset_path:
            raise ValueError("Either dataset_name or dataset_path must be specified in the config.")
        raw_dataset = load_dataset('json', data_files=dataset_path, split='train')
        logger.info(f"Dataset loaded from: {dataset_path}")

    logger.info(f"Dataset columns: {raw_dataset.column_names}")

    def format_qwen_conversation(example):
        # Start with system prompt if present
        text = f"<|im_start|>system\n{example['system']}<|im_end|>\n"
        for turn in example["conversations"]:
            role = turn["from"]
            if role == "user":
                text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
        return {"text": text}

    raw_dataset = raw_dataset.map(format_qwen_conversation, remove_columns=raw_dataset.column_names)

    max_seq_length = config.get('max_seq_length', 1024)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    logger.info("Sample formatted text:\n" + raw_dataset[0]["text"])
    tokenized = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # --- 4.1 Train/Validation Split ---
    split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train set size: {len(train_dataset)}, Eval set size: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 5. Setup Training Arguments ---
    training_args_dict = config.get('training_args')
    if not training_args_dict:
        raise ValueError("training_args must be specified in the config.")

    if 'output_dir' not in training_args_dict:
        training_args_dict['output_dir'] = '/opt/ml/model'

    training_arguments = TrainingArguments(**training_args_dict)
    logger.info(f"Training Arguments: {training_arguments}")

    # --- 6. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLoggerCallback()]
    )

    logger.info("Trainer initialized.")

    # --- 7. Train ---
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training finished.")

    # --- 8. Save Model ---
    output_dir = training_arguments.output_dir
    logger.info(f"Saving final model to {output_dir}")
    from peft import PeftModel
    if is_lora_applied and isinstance(model, PeftModel):
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        save_model_and_tokenizer(model, tokenizer, output_dir, use_lora=is_lora_applied)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    logger.info("--- SFT Training Completed Successfully ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument('--config_json', type=str, required=True,
                        help='JSON string containing the entire SFT configuration.')

    args = parser.parse_args()
    try:
        config = json.loads(args.config_json)
        print("Successfully loaded configuration from JSON string.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON configuration: {e}")
        sys.exit(1)

    logger.info(f"Configuration: {config}")
    try:
        run_sft_training(config)
    except Exception as e:
        logger.error(f"SFT training failed: {e}", exc_info=True)
        raise
