# Prebuilt GPU image for UltraBERT / GLiClass training and eval jobs (rtx-pro-6000, Blackwell; CUDA 12.8).
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages uv && uv pip install --system --break-system-packages --no-cache \
      "transformers==5.20.2" "huggingface_hub>=1.31" "gliclass==0.1.21" "flashdeberta==0.0.8" "kernels==0.17.2" \
      "datasets>=4" "polars" "pyarrow>=20" "scikit-learn" "numpy>=2" "accelerate" "einops" "trackio==0.43.0" "optuna" \
      "decision-index[transformers] @ https://github.com/apolinario/decision-index/archive/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d.tar.gz"
RUN python -c "import torch, transformers, gliclass, kernels, trackio, optuna; print(torch.__version__, transformers.__version__)"
ENV HF_HOME=/tmp/hf HF_XET_HIGH_PERFORMANCE=1
CMD ["python", "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"]
