# CAD-Coder Training Pipeline

This repository contains a two-stage training pipeline for CAD-Coder, a vision-language model that generates CadQuery code from images.

## Overview

The training is split into two stages:
1. **Stage 1**: Pre-training for feature alignment using CC3M dataset (image captioning)
2. **Stage 2**: Fine-tuning on GenCAD-Code dataset (CAD code generation)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Stage 1 Training

Train the projector layer to align vision and language features:

```bash
# Basic usage with real CC3M dataset
python stage1_training.py

# With custom parameters
python stage1_training.py --batch_size 16 --epochs 3 --learning_rate 1e-3

# With mock data for testing
python stage1_training.py --mock --mock_size 100 --batch_size 8

# Save to Hugging Face Hub
python stage1_training.py --hf_repo "your-username/cad-coder-stage1"
```

**Stage 1 Arguments:**
- `--mock`: Use mock dataset for testing
- `--mock_size`: Size of mock dataset (default: 100)
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 1)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--hf_repo`: Hugging Face repository name for saving model
- `--cache_dir`: Cache directory for datasets (default: /tmp/hf_cache)

### Stage 2 Training

Fine-tune the full model on CAD code generation:

```bash
# Basic usage with real GenCAD-Code dataset
python stage2_training.py

# Load Stage 1 checkpoint and train
python stage2_training.py --stage1_checkpoint cad_coder_stage1.pth

# With mock data for testing
python stage2_training.py --mock --mock_size 50 --batch_size 4

# Save to Hugging Face Hub
python stage2_training.py --hf_repo "your-username/cad-coder-final"
```

**Stage 2 Arguments:**
- `--mock`: Use mock dataset for testing
- `--mock_size`: Size of mock dataset (default: 50)
- `--batch_size`: Batch size (default: 4)
- `--epochs`: Number of epochs (default: 1)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--hf_repo`: Hugging Face repository name for saving model
- `--stage1_checkpoint`: Path to Stage 1 checkpoint (default: cad_coder_stage1.pth)
- `--cache_dir`: Cache directory for datasets (default: /tmp/hf_cache)

### Testing the Pipeline

Use the test script to run both stages with mock data:

```bash
# Test full pipeline
python test_pipeline.py

# Test with custom parameters
python test_pipeline.py --batch-size 2 --mock-size-stage1 20 --mock-size-stage2 10

# Test only Stage 1
python test_pipeline.py --stage1-only

# Test only Stage 2
python test_pipeline.py --stage2-only

# Test with Hugging Face upload
python test_pipeline.py --hf-repo-stage1 "your-username/test-stage1" --hf-repo-stage2 "your-username/test-stage2"
```

## Model Architecture

- **Vision Encoder**: CLIP ViT-Large/14@336px
- **Language Model**: Vicuna-7B-v1.5
- **Projector**: 2-layer MLP for feature alignment

## Datasets

- **Stage 1**: [CC3M (pixparse/cc3m-wds)](https://huggingface.co/datasets/pixparse/cc3m-wds) - ~3M image-caption pairs
- **Stage 2**: [GenCAD-Code](https://huggingface.co/datasets/CADCODER/GenCAD-Code) - CAD images with CadQuery code

## Hugging Face Integration

The scripts can automatically save models to Hugging Face Hub:

1. Login to Hugging Face:
```bash
huggingface-cli login
```

2. Create repositories on Hugging Face Hub

3. Use the `--hf_repo` argument to save models

## Output Files

- `cad_coder_stage1.pth`: Stage 1 checkpoint
- `cad_coder_stage2_final.pth`: Final model after Stage 2
- `hf_models/`: Local directory for Hugging Face uploads

## Hardware Requirements

- GPU with at least 16GB VRAM (recommended for full training)
- For testing with mock data: 8GB VRAM should be sufficient

## Notes

- The CC3M dataset uses `jpg` and `txt` fields for images and captions
- Stage 1 only trains the projector layer (vision encoder and LLM are frozen)
- Stage 2 trains both projector and LLM (vision encoder remains frozen)
- Mock datasets are automatically generated for testing purposes
