"""
Model utility functions for loading models, tokenizers, applying LoRA/PEFT, 
and handling checkpoints, compatible with SageMaker.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Tuple
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_base_model_and_tokenizer(
    model_name_or_path: str,
    use_4bit: bool = False,
    device_map: Optional[str] = 'auto',
    trust_remote_code: bool = True, # Needed for some models like Qwen
    **bnb_kwargs # Additional BitsAndBytesConfig args
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the base model and tokenizer.

    Args:
        model_name_or_path (str): Hugging Face model ID or local path.
        use_4bit (bool): Whether to load the model in 4-bit precision using BitsAndBytes.
        device_map (str): Device map strategy for model loading.
        trust_remote_code (bool): Allow execution of custom code from the model hub.
        **bnb_kwargs: Additional arguments for BitsAndBytesConfig.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer.
    """
    logger.info(f"Loading base model and tokenizer from: {model_name_or_path}")

    def find_model_dir(path: str) -> str:
        # Check for model files at root
        model_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]
        if any(os.path.isfile(os.path.join(path, f)) for f in model_files):
            return path
        # Look for a single subdirectory containing model files
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for sub in subdirs:
            sub_path = os.path.join(path, sub)
            if any(os.path.isfile(os.path.join(sub_path, f)) for f in model_files):
                logger.warning(f"Model files not found at root. Using subdirectory: {sub_path}")
                return sub_path
        raise FileNotFoundError(f"No model files found in {path} or its subdirectories.")

    bnb_config = None
    if use_4bit:
        logger.info("Using 4-bit quantization (BitsAndBytes).")
        default_bnb_config = {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_compute_dtype': torch.bfloat16,
            'bnb_4bit_use_double_quant': True,
        }
        default_bnb_config.update(bnb_kwargs) # Override defaults if provided
        bnb_config = BitsAndBytesConfig(**default_bnb_config)

    try:
        if os.path.isdir(model_name_or_path):
            # Local directory: resolve for SageMaker output structure
            resolved_model_dir = find_model_dir(model_name_or_path)
        else:
            # Hugging Face Hub model ID or non-local path
            resolved_model_dir = model_name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_dir,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16 if use_4bit else 'auto' # Match compute dtype if 4bit
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_dir,
            trust_remote_code=trust_remote_code
        )

        # Set padding token if not already set
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        logger.info("Model and tokenizer loaded successfully from %s.", resolved_model_dir)
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}", exc_info=True)
        raise

def apply_lora(
    model: AutoModelForCausalLM,
    lora_config_dict: Optional[Dict] = None,
    use_4bit: bool = False
) -> AutoModelForCausalLM:
    """
    Applies LoRA/PEFT to the model if a configuration is provided.

    Args:
        model: The base model.
        lora_config_dict (Optional[Dict]): Configuration for LoRA (PEFT).
            Example: {'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05, 
                      'bias': 'none', 'task_type': 'CAUSAL_LM', 
                      'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']}
        use_4bit (bool): Whether the base model was loaded in 4-bit.

    Returns:
        The model with PEFT adapters applied (if config provided), otherwise the original model.
    """
    if lora_config_dict is None:
        logger.info("No LoRA configuration provided. Returning base model.")
        return model

    logger.info(f"Applying LoRA with config: {lora_config_dict}")
    try:
        peft_config = LoraConfig(**lora_config_dict)
        
        # Prepare model for k-bit training if needed BEFORE applying PEFT
        if use_4bit:
            logger.info("Preparing model for k-bit training (gradient checkpointing).")
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, peft_config)
        logger.info("LoRA applied successfully.")
        model.print_trainable_parameters()
        return model
    except Exception as e:
        logger.error(f"Failed to apply LoRA: {e}", exc_info=True)
        raise

def set_special_tokens(tokenizer: AutoTokenizer, special_tokens_dict: Optional[Dict] = None) -> None:
    """
    Adds or modifies special tokens in the tokenizer.
    
    Args:
        tokenizer: The tokenizer instance.
        special_tokens_dict (Optional[Dict]): Dictionary of special tokens to add/modify.
                                             Example: {'pad_token': '<PAD>', 'additional_special_tokens': ['<THINK>', '<ANSWER>']}
    """
    if special_tokens_dict:
        logger.info(f"Setting special tokens: {special_tokens_dict}")
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        if num_added > 0:
            logger.info(f"Added {num_added} new special tokens.")
            # Important: Resize model embeddings if new tokens were added
            # model.resize_token_embeddings(len(tokenizer))
            # Note: Resizing should happen AFTER loading the model and BEFORE applying PEFT/training
            # This function should ideally be called before model loading or handled carefully.
            logger.warning("Model embeddings might need resizing if new tokens were added. Ensure this is handled before training.")

def save_model_and_tokenizer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    is_lora: bool = False
) -> None:
    """
    Saves the model and tokenizer to the specified directory.
    Handles both full model saves and LoRA adapter saves.
    Creates the directory if it doesn't exist.

    Args:
        model: The model to save (can be base or PEFT model).
        tokenizer: The tokenizer to save.
        output_dir (str): Directory to save the model and tokenizer.
        is_lora (bool): Set to True if saving PEFT adapters, False for full model.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model and tokenizer to: {output_dir}")
    
    try:
        if is_lora:
            logger.info("Saving PEFT adapters (LoRA).")
            # Saves only the adapters and config
            model.save_pretrained(output_dir)
        else:
            logger.info("Saving full model weights.")
            # Saves the full model
            model.save_pretrained(output_dir)
            
        # Always save the tokenizer
        tokenizer.save_pretrained(output_dir)
        logger.info("Model and tokenizer saved successfully.")
        
    except Exception as e:
        logger.error(f"Failed to save model/tokenizer: {e}", exc_info=True)
        raise

# --- Example Usage --- 
if __name__ == '__main__':
    # Note: Running this requires significant memory/GPU
    # This is just for demonstration of function calls
    
    test_model_name = "gpt2" # Use a small model for local testing if possible
    use_4bit_test = False # Set to True if you have GPU and bitsandbytes installed
    
    print("--- Testing load_base_model_and_tokenizer ---")
    try:
        model, tokenizer = load_base_model_and_tokenizer(test_model_name, use_4bit=use_4bit_test)
        print(f"Loaded model: {model.__class__.__name__}")
        print(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
    except Exception as e:
        print(f"Could not run basic load test: {e}")

    # --- Testing apply_lora (requires a loaded model) ---
    # print("\n--- Testing apply_lora ---")
    # lora_test_config = {
    #     'r': 4,
    #     'lora_alpha': 8,
    #     'lora_dropout': 0.01,
    #     'bias': 'none',
    #     'task_type': 'CAUSAL_LM',
    #     # Adjust target_modules based on the model (e.g., gpt2 uses 'c_attn', 'c_proj')
    #     'target_modules': ['c_attn'] 
    # }
    # try:
    #     lora_model = apply_lora(model, lora_test_config, use_4bit=use_4bit_test)
    #     print(f"LoRA model type: {lora_model.__class__.__name__}")
    # except Exception as e:
    #     print(f"Could not run LoRA apply test: {e}")
        
    # --- Testing save_model_and_tokenizer (requires a loaded model) ---
    # print("\n--- Testing save_model_and_tokenizer ---")
    # test_output_dir = "./temp_test_model_save"
    # try:
    #     save_model_and_tokenizer(model, tokenizer, test_output_dir, is_lora=False)
    #     print(f"Saved model/tokenizer to {test_output_dir}")
    #     # Clean up test directory (optional)
    #     import shutil
    #     # shutil.rmtree(test_output_dir)
    # except Exception as e:
    #     print(f"Could not run save test: {e}")

    print("\nModel utils basic structure defined.")

