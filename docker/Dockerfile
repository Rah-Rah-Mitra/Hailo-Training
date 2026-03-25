# Use Ubuntu 22.04 as the base image (x86_64)
FROM ubuntu:22.04

# Set non-interactive mode and disable .pyc files / buffering
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
      python3.10 \
      python3.10-dev \
      python3.10-venv \
      python3.10-distutils \
      python3-pip \
      python3-tk \
      build-essential \
      liblapack-dev \
      libatlas-base-dev \
      gcc \
      curl \
      git \
      graphviz \
      libgraphviz-dev \
      libgl1-mesa-glx \
      pkg-config \
      lsb-release \
      bsdmainutils \
      # Gstreamer is probably not needed. Consider removing it.
      libgstreamer1.0-0 \
      libgstreamer-plugins-base1.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory for our app
WORKDIR /app

# Create a virtual environment in /venv
# The venv is required, otherwise the Hailo Dataflow Compiler will not work properly
RUN python3.10 -m venv /venv && \
    ls -l /venv

# Set environment variables so that the venv is used
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"

# Copy the Hailo Dataflow Compiler wheel into the image and install it in the venv
COPY hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl /tmp/
RUN pip install /tmp/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl

# Switch to root temporarily to set up permissions for the SDK client directory
USER root
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# Not sure if this next line is still needed. Maybe can be removed.
RUN mkdir -p /usr/local/lib/python3.10/sdk_client && chmod -R 777 /usr/local/lib/python3.10/sdk_client
RUN chown -R appuser /venv

# Install PyTorch with CUDA support (needed by ultralytics for GPU finetuning)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy your requirements file and install Python dependencies in the venv
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

USER appuser


# Not sure if this is still needed. 
RUN ln -sf /venv/bin/python3 /venv/bin/python

# Default to an interactive shell; the venv is already activated via PATH.
CMD ["bash"]