# CUDA-enabled base with PyTorch preinstalled to avoid re-downloading GPU wheels
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
# Install torch/torchvision wheels that match CUDA. Override here if a specific
# torch version is desired for the installed CUDA runtime.
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY modnet_service ./modnet_service

EXPOSE 8000

CMD ["uvicorn", "modnet_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
