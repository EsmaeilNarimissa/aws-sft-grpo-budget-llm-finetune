# === Base image: Minimal Python 3.10 (no Conda interference)
FROM python:3.10-slim

# === Set working directory for SageMaker compatibility
WORKDIR /opt/ml/code

# === Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# === Upgrade pip & core build tools
RUN pip install --upgrade pip setuptools wheel

# === Install PyTorch with CUDA 11.8 support (for ml.g5 instances)
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# === Install NumPy first to avoid any post-install overwrite from other packages
RUN pip install numpy==1.26.4

# === Install ML packages and SageMaker runtime
RUN pip install \
    transformers==4.51.3 \
    trl==0.16.1 \
    peft==0.7.1 \
    accelerate==1.6.0 \
    datasets==3.5.0 \
    sagemaker-training \
    boto3 \
    awscli \
    evaluate \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    regex \
    tqdm \
    ipykernel \
    python-dotenv \
    tensorboard \
    ipywidgets \
    antlr4-python3-runtime==4.13.2 \
    latex2sympy2-extended==1.10.1 \
    math_verify==0.7.0

# === Verify Transformers Version ===
RUN python -c "import transformers; print('Transformers version installed in image:', transformers.__version__)"
