# Reasoning on a Budget: Reproducing DeepSeek R1 with SFT + GRPO on AWS (Qwen2.5B)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange)

> **TL;DR:**  
> Fine-tune an instruction-tuned LLM (Qwen2.5B) on reasoning tasks using a DeepSeek R1-inspired pipeline (SFT + GRPO).  
> Achieve competitive alignment using rule-based rewards — all for under $100 on AWS SageMaker.

> **Based on the arXiv paper:**  
> **Reasoning on a Budget: Miniaturizing DeepSeek R1 with SFT-GRPO Alignment for Instruction-Tuned LLMs**  
> 📌 **[arXiv preprint coming soon – check back for DOI link]**

> **📖 Read the Full Article**
>
> For a detailed description of the methodology, experiments, and results, please read our companion article (coming soon on arXiv):
>
> **Reasoning on a Budget: Miniaturizing DeepSeek R1 with SFT-GRPO Alignment for Instruction-Tuned LLMs**
>
> _A future update will include the direct arXiv link here._
>
> The article provides:
> - Theoretical background and motivation
> - Full details of the SFT and GRPO training stages
> - Reward function design and prompting conventions
> - Experimental results and analysis
> - Discussion of limitations and future work
>
> **We recommend starting with the article for a comprehensive understanding of the codebase, design decisions, and how to reproduce or build upon our results.**

---

## Pretrained Model

The trained model (after SFT and GRPO) is available on Hugging Face Hub:

[Essi-Narim/qwen2.5b-sft-grpo-reasoning](https://huggingface.co/Essi-Narim/qwen2.5b-sft-grpo-reasoning/tree/main)

You can load the model directly with Hugging Face Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Essi-Narim/qwen2.5b-sft-grpo-reasoning")
tokenizer = AutoTokenizer.from_pretrained("Essi-Narim/qwen2.5b-sft-grpo-reasoning")
```

---

This project implements a modular DeepSeek R1-style training pipeline for reasoning-centric language models, using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). The pipeline is designed for both local development (VS Code, Jupyter) and scalable training on AWS SageMaker.

---

## Project Structure

```
.
├── notebooks/
│   └── main_controller.ipynb # Orchestration notebook
├── src/
│   ├── model_utils.py       # Model/tokenizer utilities
│   ├── reward_functions.py  # Composable, testable reward functions
│   ├── train_grpo.py        # GRPO training script (SageMaker compatible)
│   └── train_sft.py         # SFT training script (SageMaker compatible)
├── grpo-sagemaker-image/
│   └── Dockerfile           # SageMaker-compatible Docker image
├── requirements.txt         # Python package dependencies
├── report/                  # (Optional) Markdown reports, plots, analysis
```

---

## Example Configuration Dictionaries

Below are **example** (sanitized) config dictionaries for SFT and GRPO. Replace all placeholders (`<your-bucket>`, `<your-dataset-name>`, etc.) with your actual values. **Never commit real AWS account IDs, S3 buckets, or credentials.**

```python
sft_config = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset_name": "<your-dataset-name>",
    "max_seq_length": 1024,
    "s3_output_path": "s3://<your-bucket>/<your-path>/sft_model/",
    "use_lora": True,
    "lora_config": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
    },
    "training_args": {
        "output_dir": "/opt/ml/model",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "logging_steps": 10,
        "save_steps": 50
    }
}

# ...

grpo_config = {
    "base_model_path": "/opt/ml/input/data/sft_model",
    "dataset_name": "<your-dataset-name>",
    "dataset_prompt_field": "prompt",
    "s3_output_path": "s3://<your-bucket>/<your-path>/grpo_model/",
    "reward_function_weights": {
        "accuracy_cosine_scaled": 0.5,
        "format": 0.1,
        "reasoning": 0.15,
        "conciseness": 0.05,
        "repetition_penalty": 0.2
    },
    "use_lora": True,
    "training_args": {
        "output_dir": "/opt/ml/model",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-6,
        "logging_steps": 10,
        "save_steps": 50
    },
    "grpo_args": {
        "num_iterations": 3,
        "num_generations": 2,
        "beta": 0.1,
        "max_prompt_length": 512,
        "max_completion_length": 512
    }
}
```

---

## Building & Using the SageMaker Docker Image

This project provides a custom Dockerfile for AWS SageMaker training in `grpo-sagemaker-image/Dockerfile`.

**Quick Steps:**

1. **Build the Docker image locally:**
   ```bash
   cd grpo-sagemaker-image
   docker build -t grpo-sagemaker .
   ```
2. **Test the image locally (optional, recommended):**
   ```bash
   docker run -it grpo-sagemaker bash
   # Inside the container, test imports or run scripts, e.g.:
   python -c "import torch; import transformers; import trl; print('All good!')"
   ```
3. **Push to AWS ECR for SageMaker:**
   - Create an ECR repository (if needed):
     ```bash
     aws ecr create-repository --repository-name grpo-sagemaker --region <your-region>
     ```
   - Authenticate Docker to ECR:
     ```bash
     aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your_aws_account_id>.dkr.ecr.<your-region>.amazonaws.com
     ```
   - Tag and push your image:
     ```bash
     docker tag grpo-sagemaker:latest <your_aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/grpo-sagemaker:latest
     docker push <your_aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/grpo-sagemaker:latest
     ```

**Full step-by-step instructions, including Windows CMD commands and troubleshooting, are provided in [`Docker_image_SM.ipynb`](./Docker_image_SM.ipynb).**


## Training Stages

The pipeline consists of two main training phases:

1.  **Phase 1: Supervised Fine-Tuning (SFT)**
    *   **Script:** `src/train_sft.py`
    *   **Goal:** Adapt a pre-trained base language model to your domain or task using prompt-completion pairs. LoRA (Low-Rank Adaptation) is optionally supported for parameter-efficient fine-tuning.
    *   **Note:** All completions must be formatted with `<think>...</think>` and `<answer>...</answer>` tags, as required by the project rules.

2.  **Phase 2: Group Relative Policy Optimization (GRPO)**
    *   **Script:** `src/train_grpo.py`
    *   **Goal:** Further refine the SFT model using preference-based RL, guided by custom reward functions. Multiple completions are compared and scored. The `trl` library's `GRPOTrainer` is used for optimization.
    *   **Reward Functions:** Implemented in `src/reward_functions.py`, all reward functions are composable and testable individually. Use the provided `combine_rewards()` utility to create composite rewards for training.

## Environment Setup

> **Note:** This codebase is designed for modularity and SageMaker compatibility. All training scripts avoid Jupyter-only constructs and support S3 paths for outputs and checkpoints. Local development is supported via VS Code and the orchestration notebook.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) AWS Configuration for SageMaker:**
    *   Install AWS CLI: [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
    *   Configure AWS credentials:
        ```bash
        aws configure
        ```
        (You'll need your Access Key ID, Secret Access Key, default region, and output format). Ensure the associated IAM user/role has permissions for SageMaker, S3, IAM, etc.
    *   Install `boto3` and `sagemaker` (already included in `requirements.txt`).

## Running Training Stages

### Option 1: Using the Orchestration Notebook (Local Execution)

The `notebooks/main_controller.ipynb` provides a structured way to run the SFT and GRPO stages locally (ideal for testing and debugging on smaller datasets/models).

1.  Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
2.  Open `notebooks/main_controller.ipynb`.
3.  Configure the `sft_config` and `grpo_config` dictionaries within the notebook cells according to your local paths, model choices, and desired training parameters.
4.  Execute the cells sequentially to run the SFT and then the GRPO training stages.

### Option 2: Using Command-Line Scripts (Local or SageMaker Preparation)

The `src/train_sft.py` and `src/train_grpo.py` scripts can be run directly from the command line. They accept arguments for configuration, making them suitable for local runs or as entry points for SageMaker jobs.

**Example: Running SFT locally via CLI:**

```bash
python src/train_sft.py \
    --model_name_or_path "gpt2" \
    --dataset_path "data/your_sft_data.jsonl" \
    --output_dir "results/local_sft_run" \
    --use_lora "False" \
    --training_args '{"num_train_epochs": 1, "per_device_train_batch_size": 2, "logging_steps": 10, "save_steps": 50, "learning_rate": 5e-5}' \
    # Add other required args like --max_seq_length, --dataset_text_field etc.
```

**Example: Running GRPO locally via CLI (after SFT):**

```bash
python src/train_grpo.py \
    --base_model_path "results/local_sft_run" \
    --dataset_path "data/your_grpo_prompts.jsonl" \
    --output_dir "results/local_grpo_run" \
    --use_lora "False" \
    --reward_weights '{"accuracy": 0.5, "format": 0.5}' \
    --grpo_config '{"beta": 0.1, "num_train_epochs": 1, "batch_size": 1, "max_prompt_length": 128, "max_completion_length": 256, "learning_rate": 1e-5, "logging_steps": 5}' \
    # Add other required args like --dataset_prompt_field etc.
```

*(Note: Replace placeholders like paths, model names, and JSON config strings with your actual values.)*

## Launching SageMaker Jobs

> **Note:** There is no `scripts/` directory or `launch_grpo_job.py` in this codebase. To launch SageMaker jobs, use the SageMaker Python SDK and point to the appropriate entry script in `src/` (see PRD.md for guidance). Ensure your Docker image is built from `grpo-sagemaker-image/Dockerfile` if using a custom environment.

**Usage:**

```bash
python scripts/launch_grpo_job.py --help # To see all available arguments

# Example: Launching a GRPO Training Job on SageMaker
python scripts/launch_grpo_job.py \
    --job-name-prefix "my-grpo-job" \
    --iam-role "arn:aws:iam::YOUR_ACCOUNT_ID:role/YourSageMakerExecutionRole" \
    --s3-input-data "s3://your-bucket/path/to/grpo_prompts.jsonl" \
    --s3-output-path "s3://your-bucket/output/grpo_training" \
    --entry-point "train_grpo.py" \
    --source-dir "../src" \
    --instance-type "ml.g4dn.xlarge" \
    --hyperparameters '{
        "base_model_path": "/opt/ml/input/data/sft_model", 
        "dataset_path": "/opt/ml/input/data/train/grpo_prompts.jsonl", 
        "output_dir": "/opt/ml/model", 
        "tokenizer_path": "/opt/ml/input/data/sft_model", 
        "use_lora": "False", 
        "reward_weights": "{\\"accuracy\\": 0.5, \\"format\\": 0.5}", 
        "grpo_config": "{\\"beta\\": 0.1, \\"num_train_epochs\\": 1, \\"batch_size\\": 1, \\"max_prompt_length\\": 128, \\"max_completion_length\\": 256, \\"learning_rate\\": 1e-5, \\"logging_steps\\": 10}",
        "dataset_prompt_field": "prompt"
    }' \
    # Add other args like --instance-count, --volume-size, framework versions if needed
```

**Important Notes for SageMaker:**

*   Replace placeholders (Account ID, Role Name, S3 paths, instance type) with your actual values.
*   The `--hyperparameters` argument must be a valid JSON string. **Crucially**, paths inside this JSON (`base_model_path`, `dataset_path`, `output_dir`) **must** refer to the expected paths *inside* the SageMaker container (e.g., `/opt/ml/input/data/train`, `/opt/ml/model`).
*   Ensure the input data (`--s3-input-data`) and any required base models (if not loading from Hugging Face Hub) are correctly placed in S3 and accessible by the SageMaker execution role. You might need multiple input channels if providing a base model from S3.

## Example Outputs (Placeholder)

*This section will be updated after running the pipeline.*

- Example SFT and GRPO completions and reward logs will be provided once available.
## Known Limitations & Future Extensions

*   **Dataset Dependency:** The quality and nature of the SFT and preference datasets heavily influence the final model performance.
*   **Reward Function Complexity:** Designing effective reward functions that accurately capture desired reasoning traits can be challenging.
*   **Hyperparameter Tuning:** Optimal hyperparameters (learning rates, batch sizes, LoRA config, GRPO beta, etc.) may require significant tuning.
*   **Scalability:** Training very large models may require multi-GPU or distributed training setups beyond the current single-instance configuration.

**Possible Extensions:**

*   Implement more sophisticated reward functions (e.g., using a separate reward model).
*   Integrate techniques like Direct Preference Optimization (DPO).
*   Add robust evaluation metrics and test sets specific to reasoning tasks.
*   Explore different base models.
*   Implement distributed training support.

---

## Contact

For questions, feedback, or collaboration inquiries, please contact:

**Esmaeil Narimissa**  
esmaeil.narimissa@gmail.com
