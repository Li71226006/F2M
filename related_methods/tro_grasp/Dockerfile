FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    ffmpeg \
    libegl1 \
    libgl1 \
    libglfw3 \
    libgles2 \
    libosmesa6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/external/TRO-Grasp

# Build with repository root as context:
# docker build -f F2M/related_methods/tro_grasp/Dockerfile -t f2m-tro-grasp:local .
COPY external/TRO-Grasp/requirements.txt /tmp/tro_requirements.txt
RUN python3 -m pip install -r /tmp/tro_requirements.txt

ENV PYTHONPATH=/workspace:/workspace/src:/workspace/external/TRO-Grasp:/workspace/F2M

CMD ["python3", "render_mujoco_sparse_grasps.py", "--object", "ycb+baseball", "--modes", "4f_no_little", "--output-dir", "/workspace/F2M/results/related_methods/tro_grasp", "--source", "dataset"]
