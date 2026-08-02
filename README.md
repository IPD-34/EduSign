# Branch: feature/backbone-training-pipeline

## Overview

This branch implements the backbone recognition pipeline for **EduSign**, covering the complete Month 1 scaffold and the initial Week 2 infrastructure for the recognition transformer.

The work establishes the end-to-end preprocessing, landmark extraction, dataset preparation, transformer backbone, dataloading, and training framework required before full ASLLRP training.

---

## Implemented Features

### Dataset Acquisition & Preparation

- WLASL acquisition pipeline
- WLASL dataset inspection and validation utilities
- Gloss coverage analysis
- Missing download recovery and resume support
- Vocabulary generation from processed datasets
- Hugging Face mirror support (reader and merge utilities)

### ASLLRP Support

- XML metadata reader
- Frame indexing verification utilities
- Sequence length analysis
- Frame difference inspection
- Parallel preprocessing pipeline
- Boundary verification scripts

---

### MediaPipe Landmark Pipeline

- Holistic landmark extraction
- Pose detector
- Hand detector
- Face detector
- Pose-guided face ROI extraction
- Face landmark remapping
- Unified landmark schema
- Frame assembly pipeline

---

### Dataset Preprocessing

- Reader-agnostic preprocessing pipeline
- Video reader
- Frame sampling
- Sequence builder
- Dataset processor
- Sample schema
- Sample serialization
- Metadata readers for:
  - WLASL
  - ASLLRP
  - HF Mirror

---

### Transformer Backbone

Implemented a baseline Transformer architecture using MediaPipe Holistic landmarks.

Components include:

- Feature flattening utilities
- Positional Encoding
- Transformer Encoder
- Transformer Decoder
- End-to-end Sign Transformer

Architecture constraints followed:

- MediaPipe landmarks only
- No raw RGB input
- No CNN
- No GCN
- Full self-attention Transformer

---

### Dataset & Training Infrastructure

- PyTorch Dataset implementation
- Variable-length batching
- Padding and attention masks
- Teacher-forced training utilities
- Optimizer & scheduler
- Checkpoint save/load
- Trial training script

---

### Testing & Validation

Implemented validation scripts for:

- Landmark detectors
- Dataset processor
- Video reader
- Metadata readers
- Sequence builder
- Dataloader
- Backbone smoke tests
- Training loop verification
- Visualization utilities

---

## Current Status

### Completed

- WLASL preprocessing pipeline
- ASLLRP preprocessing framework
- Holistic landmark extraction pipeline
- Dataset preprocessing framework
- Transformer backbone scaffold
- Variable-length dataloader
- Initial training loop
- Smoke-tested forward/backward propagation
- Trial training validation
- Vocabulary generation

### In Progress

- Full ASLLRP preprocessing
- GPU device integration
- Full-scale backbone training
- Month 2 end-to-end CTC + Boundary Detection training

---

## Project Structure

```
scripts/
    wlasl_acquisition/
    asllrp_acquisition/
    preprocessing runners
    training
    testing

src/
    landmarks/
    preprocessing/
    models/
    visualization/
```

---

## Tech Stack

- Python
- PyTorch
- MediaPipe Holistic
- OpenCV
- NumPy
- WLASL
- ASLLRP
- UV

---

## Branch Scope

This branch establishes the complete backbone training infrastructure for the EduSign recognition model and provides the foundation required for subsequent end-to-end training, evaluation, and deployment.