# Satellite-Guided DDColor Enhancement

This project improves frozen DDColor street-view colorization with global polar satellite context.

Core pipeline:

```text
grayscale panorama patch
  -> frozen DDColor
  -> base RGB result
  -> FiLM-conditioned residual color correction
  -> final RGB result

polar satellite image
  -> ResNet18 polar context encoder
  -> global context vector
```

DDColor is used only as a frozen base colorizer. The trainable modules are:

- Polar Context Encoder
- FiLM conditioning layers
- Residual Color Correction module

## Features

- Frozen official DDColor integration
- Polar-context-guided color correction
- Four-patch panorama inference
- Mixed precision GPU training
- Checkpoint save/resume
- PSNR, SSIM, LPIPS, and FID evaluation
- Minimal Gradio inference UI

## Data Layout

Expected processed CVUSA-style layout:

```text
CVUSA_processed_split/
  train/
    ground_rgb/
    ground_gray/
    overhead_polar/
    overhead_polar_seg/
  val/
    ground_rgb/
    ground_gray/
    overhead_polar/
    overhead_polar_seg/
  test/
    ground_rgb/
    ground_gray/
    overhead_polar/
    overhead_polar_seg/
```

The dataset loader keeps complete panorama groups only. A valid panorama must have four matched street-view patches sharing the same polar context image.

Raw images are read only and are never overwritten.

## Setup

Create a Python environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

For RTX 50-series GPUs, install PyTorch wheels with `sm_120` support:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Install the remaining dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))"
```

## Configuration

Copy the example config and edit paths:

```powershell
Copy-Item config.example.yaml config.yaml
```

Set these fields in `config.yaml`:

```yaml
dataset:
  root: "PATH/TO/CVUSA_processed_split"

ddcolor:
  code_path: "PATH/TO/DDColor"
  weights_path: "PATH/TO/DDColor/weights_hf/ddcolor_paper_tiny/pytorch_model.bin"
```

The local `config.yaml` is ignored by Git so machine-specific paths are not committed.

## Commands

Run a data smoke test:

```powershell
.\.venv\Scripts\python.exe train.py --smoke_test
```

Run a very short end-to-end GPU training check:

```powershell
.\.venv\Scripts\python.exe train.py --quick_train --exp_name smoke_gpu
```

Train:

```powershell
.\.venv\Scripts\python.exe -u train.py --exp_name film_ddcolor
```

Resume:

```powershell
.\.venv\Scripts\python.exe train.py --resume checkpoints\film_ddcolor\latest.pth --exp_name film_ddcolor
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints\film_ddcolor\best.pth --split val
```

Run panorama inference:

```powershell
.\.venv\Scripts\python.exe inference.py --checkpoint checkpoints\film_ddcolor\best.pth --street_view path\street.jpg --satellite path\satellite.jpg --output outputs\result.jpg --show_base
```

Launch the minimal UI:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://localhost:7861
```

The current UI accepts:

- panorama street-view image
- matching polar image

and outputs:

- final enhanced result
- frozen DDColor base result
- comparison image

## Checkpoints

The trained checkpoint is stored with Git LFS:

```text
checkpoints/film_ddcolor_cu130_20260527/best.pth
```

Install Git LFS before cloning or pulling the checkpoint:

```powershell
git lfs install
git clone https://github.com/chencengcheng2-cmd/satellite_guided_ddcolor.git
```

If the repository has already been cloned without LFS objects, run:

```powershell
git lfs pull
```

## Results From Local Run

A local 30-epoch validation run achieved:

| Metric | Frozen DDColor | Satellite-Guided |
| --- | ---: | ---: |
| PSNR | 23.9258 | 29.0234 |
| SSIM | 0.9570 | 0.9814 |
| LPIPS | 0.1604 | 0.1121 |
| FID | 15.1953 | 12.6313 |

## Notes

- DDColor parameters are frozen and excluded from the optimizer.
- Polar images are used as global context; no pixel-level alignment is assumed.
- One panorama's four patches share the same polar context image.
- The default polar input size is `256 x 512`.
- LPIPS and FID are available in `evaluate.py`; per-epoch validation keeps them disabled by default for speed.
