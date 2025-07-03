# use a Docker image with CUDA-compatible PyTorch
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime


# environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# create working directory
WORKDIR /workspace

# install dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# add entrypoint for code and training run
COPY entrypoint.sh ./entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]