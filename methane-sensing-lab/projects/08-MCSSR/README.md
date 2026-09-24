# AI Methane Detection

An AI-based methane plume detection template using satellite imagery (e.g., Sentinel-2, PRISMA, EMIT).
This repository is a minimal, runnable scaffold for training and inference.

## Features
- Lightweight U-Net-based detection model
- Clear train/infer pipelines with configs
- Ready for CI and testing
- Export-friendly structure (add ONNX/TensorRT as needed)

## Quick Start
```bash
# 1) Create env (conda recommended)
conda env create -f environment.yml
conda activate ai-methane-detection

# 2) Train (dummy dataset for smoke test)
python src/train.py --config configs/train.yaml

# 3) Inference (uses the saved checkpoint)
python src/infer.py --config configs/infer.yaml
```

## Project Structure
```text
ai-methane-detection/
├─ configs/           # YAML configs for train/infer
├─ data/              # raw/processed data (placeholders)
├─ outputs/           # checkpoints, logs, results
├─ scripts/           # helper scripts
├─ src/               # core code: model, train, infer, utils
├─ tests/             # unit tests
└─ .github/workflows/ # CI
```

## Notes
- This template uses a dummy dataset for quick verification.
- Replace `DummyDataset` and data loaders with your real pipeline.
- Add augmentations, metrics, logging (e.g., TensorBoard/W&B) as needed.
