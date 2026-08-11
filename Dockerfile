ARG CUDA_IMAGE=docker.m.daocloud.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG ONNXRUNTIME_GPU_VERSION=1.23.2
ARG YOLOX_MODEL_URL=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
ARG YOLOX_MODEL_SHA256=427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7

ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    UV_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:${PATH} \
    PERSON_SEARCH_HOST=0.0.0.0 \
    PERSON_SEARCH_PORT=8000 \
    PERSON_SEARCH_PREFER_CUDA=true \
    PERSON_SEARCH_YOLOX_MODEL=/app/models/yolox_tiny.onnx \
    PERSON_SEARCH_INSIGHTFACE_ROOT=/models/.insightface

WORKDIR /app

# Ubuntu 22.04 provides Python 3.10, which is supported by this project.
RUN sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
      -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
      /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      python3.10 \
      python3.10-venv \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" "uv==0.6.9"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# The lock file contains CPU onnxruntime for the inference-cpu extra. Replace
# that package with the GPU build after syncing the remaining dependencies.
RUN uv sync --frozen --python /usr/bin/python3.10 --extra inference-cpu \
    && uv pip uninstall --python /app/.venv/bin/python onnxruntime \
    && uv pip install --python /app/.venv/bin/python \
      --index-url "${PIP_INDEX_URL}" "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}"

RUN mkdir -p /app/models \
    && curl --fail --location --retry 3 "${YOLOX_MODEL_URL}" \
      --output /app/models/yolox_tiny.onnx \
    && echo "${YOLOX_MODEL_SHA256}  /app/models/yolox_tiny.onnx" | sha256sum --check --strict

RUN mkdir -p /models/.insightface

VOLUME ["/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:8000/healthz || exit 1

CMD ["person-search-api"]
