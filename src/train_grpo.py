import logging
import os
import json
import argparse
from typing import Dict, List, Any
import random
import numpy as np
import torch
import shutil
import pandas as pd
import tarfile

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, TrainerCallback
from transformers.utils import is_torch_xla_available
from trl import GRPOTrainer, GRPOConfig

from model_utils import (
    load_base_model_and_tokenizer,
    apply_lora,
    save_model_and_tokenizer
)
import reward_functions as rf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RewardLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.reward_logs = []

    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if logs and "reward/mean" in logs:
            self.reward_logs.append({
                'step': state.global_step,
                'epoch': state.epoch,
                'mean_reward': logs.get('reward/mean', 0.0)
            })

    def on_train_end(self, args, state, control, **kwargs):
        if self.reward_logs:
            log_path = os.path.join(args.output_dir, 'reward_logs.csv')
            try:
                pd.DataFrame(self.reward_logs).to_csv(log_path, index=False)
                logger.info(f"Saved reward logs to {log_path}")
            except Exception as e:
                logger.error(f"Failed to save reward logs: {e}")
        else:
            logger.warning("No reward logs were collected.")

class CustomGRPOTrainer(GRPOTrainer):
    def _load_rng_state(self, resume_from_checkpoint):
        rng_file = os.path.join(resume_from_checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            logger.warning(f"Didn't find RNG state at {rng_file}")
            return
        try:
            checkpoint_rng_state = torch.load(rng_file, weights_only=True)
        except Exception:
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)

        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available() and "cuda" in checkpoint_rng_state:
            cuda_state = checkpoint_rng_state["cuda"]
            if isinstance(cuda_state, list) and cuda_state:
                torch.cuda.set_rng_state(cuda_state[0], device=0)
            elif isinstance(cuda_state, torch.Tensor):
                torch.cuda.set_rng_state(cuda_state, device=0)

        if is_torch_xla_available() and "xla" in checkpoint_rng_state:
            import torch_xla.core.xla_model as xm
            xm.set_rng_state(checkpoint_rng_state["xla"])


def run_grpo_training(config: Dict) -> None:
    logger.info("--- Starting GRPO Training ---")
    logger.info(f"Configuration: {config}")

    base_output_dir = os.environ.get("SM_CHANNEL_CHECKPOINT", config.get('training_args', {}).get('output_dir', './grpo_results'))
    output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    config_dump_path = os.path.join(output_dir, "config_used.json")
    with open(config_dump_path, "w") as f:
        json.dump(config, f, indent=2)

    config['output_dir'] = output_dir

    base_model_path = os.environ.get("SM_CHANNEL_SFT_MODEL")
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"SFT model path {base_model_path} not found.")

    for item in os.listdir(base_model_path):
        if item.endswith(".tar.gz"):
            tarfile.open(os.path.join(base_model_path, item), "r:gz").extractall(base_model_path)

    model, tokenizer = load_base_model_and_tokenizer(
        model_name_or_path=base_model_path,
        use_4bit=config.get('use_4bit', False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    use_lora_saving = config.get('use_lora', False)
    peft_config_grpo = config.get('lora_config')

    dataset = load_dataset(config['dataset_name'])
    def preprocess_function(example):
        if "messages" in example and isinstance(example["messages"], list):
            user_prompt = example["messages"][0].get("content", "")
            assistant_response = example["messages"][1].get("content", "")
        else:
            user_prompt = example.get(config['dataset_prompt_field'], "")
            assistant_response = example.get("solution", "")
        return {"prompt": user_prompt, "solution": assistant_response}

    processed_train_dataset = dataset['train'].map(preprocess_function, remove_columns=list(dataset['train'].features.keys()))

    reward_weights = config.get('reward_weights', config.get('reward_function_weights', {}))
    def compute_rewards(prompts: List[str], completions: List[str], **kwargs) -> List[torch.Tensor]:
        labels = kwargs.get('solution')
        rewards = []
        for i, output in enumerate(completions):
            accuracy = rf.accuracy_reward(output, labels[i]) if labels else 0.0
            scaled_accuracy = rf.cosine_scaled_reward(accuracy, output)
            format_val = rf.format_reward(output)
            reasoning = rf.reasoning_steps_reward(output)
            conciseness = rf.conciseness_reward(output)
            repetition = rf.repetition_penalty_reward(output)
            score = rf.combine_rewards({
                'accuracy_cosine_scaled': scaled_accuracy,
                'format': format_val,
                'reasoning': reasoning,
                'conciseness': conciseness,
                'repetition_penalty': repetition
            }, reward_weights)
            rewards.append(torch.tensor(score, dtype=torch.float32))
        return rewards

    # Merge both training_args and grpo_args before filtering for valid GRPOConfig keys
    combined_args = {}
    combined_args.update(config.get('training_args', {}))
    combined_args.update(config.get('grpo_args', {}))  # This ensures num_generations and other GRPO-specific params are included
    valid_keys = GRPOConfig.__init__.__code__.co_varnames
    filtered_args = {k: v for k, v in combined_args.items() if k in valid_keys}
    grpo_args = GRPOConfig(**filtered_args)

    trainer = CustomGRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=compute_rewards,
        train_dataset=processed_train_dataset,
        peft_config=peft_config_grpo,
        callbacks=[RewardLoggingCallback()],
        processing_class=None
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    save_model_and_tokenizer(
        model=trainer.model,
        tokenizer=tokenizer,
        output_dir=os.path.join(output_dir, "final_model"),
        is_lora=use_lora_saving
    )

    # Copy reward logs first
    reward_log_src = os.path.join(output_dir, "reward_logs.csv")
    sagemaker_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    if os.path.exists(reward_log_src):
        shutil.copy2(reward_log_src, sagemaker_model_dir)
        logger.info(f"✅ Copied reward_logs.csv to {sagemaker_model_dir}")
    else:
        logger.warning("⚠️ reward_logs.csv not found, skipping copy.")

    final_model_dir = os.path.join(output_dir, "final_model")
    if os.path.exists(final_model_dir):
        shutil.copytree(final_model_dir, sagemaker_model_dir, dirs_exist_ok=True)
        logger.info(f"✅ Copied final model to SageMaker model directory: {sagemaker_model_dir}")
    else:
        logger.warning(f"⚠️ Final model directory {final_model_dir} not found. Model copy skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO Training")
    parser.add_argument('--config_json', type=str, required=True, help='JSON string containing the entire GRPO configuration.')
    args = parser.parse_args()

    config = json.loads(args.config_json)
    logger.info(f"Received GRPO configuration: {config}")
    run_grpo_training(config)
