# Hailo DFC Development Container for YOLO26

Docker-based development environment for finetuning YOLO26 models and compiling them to `.hef` files using the Hailo Dataflow Compiler (DFC) for Hailo-8 deployment.

## Prerequisites

### Host Machine
- **OS**: Windows 10/11 with WSL2 backend
- **GPU**: NVIDIA GPU with driver 525+ installed on Windows
- **RAM**: 16 GB minimum, 32 GB recommended (DFC compilation is memory-intensive)
- **Docker Desktop**: Installed with WSL2 backend enabled

### NVIDIA Container Toolkit
Required for GPU passthrough into the container. Install inside your WSL2 distro:

```bash
# In WSL2 terminal:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access works:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Hailo Files (must be in project root)
Download these from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/):
- `hailo_dataflow_compiler-3.XX.X-py3-none-linux_x86_64.whl` — DFC Python wheel (version may vary)
- `hailort_4.XX.X_amd64.deb` or `hailort_4.XX.X_arm64.deb` — Hailo Runtime package

See "How to Build the Docker Image" section below for detailed instructions on downloading these files.

## Project Structure

```
Hailo/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── README.md
├── hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
└── hailort_4.XX.X_amd64.deb (or arm64)
```

## Installed Packages

### System (apt)
| Package | Purpose |
|---------|---------|
| python3.10, python3.10-dev, python3.10-venv | Python runtime + venv (required by DFC) |
| build-essential, gcc, liblapack-dev, libatlas-base-dev | C/C++ compilation for native Python packages |
| graphviz, libgraphviz-dev | Model graph visualization |
| libgl1-mesa-glx | OpenCV headless rendering |
| curl, git | General utilities |

### Python (pip — installed in `/venv`)
| Package | Version | Purpose |
|---------|---------|---------|
| hailo_dataflow_compiler | 3.33.1 | Hailo DFC — parses, quantizes, and compiles models to `.hef` |
| torch | 2.5.1 (CUDA 12.4) | PyTorch with GPU support for training/finetuning |
| torchvision | 0.20.1 (CUDA 12.4) | Image transforms and pretrained model utilities |
| ultralytics | >=8.4.20 | YOLO26 model loading, training, export |
| onnx | 1.16.0 | ONNX model format support |
| onnxsim | 0.4.36 | ONNX graph simplification |
| opencv-python | 4.11.0.86 | Image processing |
| pillow | 11.1.0 | Image I/O |
| numpy | 1.26.4 | Array operations |
| seaborn | latest | Data visualization and statistical plotting |
| datasets | latest | Hugging Face datasets library for working with datasets |

> The DFC wheel also pulls in its own dependencies (TensorFlow 2.18.0, onnxruntime, etc.) automatically.

## How to Build the Docker Image

Hailo does not put anything in public registries. This means you cannot use things like PyPI or DockerHub to install dependencies directly.

### Step 1: Download Hailo Components

To get this working, you need to:

1. Go to [Hailo's Developer Zone](https://hailo.ai/developer-zone/software-downloads/)
2. Create an account if you don't have one
3. Download the **Hailo Dataflow Compiler wheel**:
   - File name: `hailo_dataflow_compiler-3.XX.X-py3-none-linux_x86_64.whl` (version may vary)
   - Place it in your project directory at the same level as the Dockerfile

4. Download the **Hailo Runtime deb package**:
   - File name: `hailort_4.XX.X_amd64.deb` or `hailort_4.XX.X_arm64.deb` (depending on your architecture)
   - Place it in your project directory at the same level as the Dockerfile

### Step 2: Verify Files Are in Place

Your project directory should now look like:
```
Hailo/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── README.md
├── hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
└── hailort_4.XX.X_amd64.deb
```

### Step 3: Build and Run

With the Hailo files in place, build the Docker image:

```bash
docker-compose up -d --build
```

The `--build` flag ensures the image is rebuilt with the latest requirements.txt and Hailo components.

### Running the Container

Start the container in detached mode:

```bash
docker-compose up -d --build
```

Then, enter the running container:

```bash
docker exec -it hailo-app-run bash
```

Or, if you prefer a more interactive approach, use:

```bash
docker-compose run app
```

### Verify the Setup

Inside the container, verify everything is working:

```bash
# Check GPU is visible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Check DFC is installed
python -c "import hailo_sdk_client; print('Hailo DFC OK')"

# Check YOLO26 loads
python -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); print('YOLO26n OK')"

# Check seaborn and datasets are installed
python -c "import seaborn; import datasets; print('seaborn and datasets OK')"
```

## Usage Examples

All commands below are run **inside the container**.

### Finetune YOLO26n on a custom dataset

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="path/to/data.yaml", epochs=50, imgsz=640, device=0)
```

### Export to ONNX

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # your finetuned weights
model.export(format="onnx", opset=11, imgsz=640)
```

### Compile to HEF (Hailo-8)

```python
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8")
hn, npz = runner.translate_onnx_model("best.onnx", net_name="yolo26n_custom")

# Quantize with calibration images (numpy array, NCHW, float32, 0-1 range)
runner.optimize(calibration_dataset)

# Compile
hef = runner.compile()
with open("yolo26n_custom.hef", "wb") as f:
    f.write(hef)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker: Error response from daemon: could not select device driver` | Install NVIDIA Container Toolkit (see Prerequisites) |
| `torch.cuda.is_available()` returns `False` | Ensure NVIDIA driver is 525+ on host, restart Docker Desktop |
| DFC compilation killed / OOM | Increase Docker Desktop memory to 16GB+ (Settings > Resources) |
| `ModuleNotFoundError: ultralytics` or similar | You may be outside the venv — run `source /venv/bin/activate` |
