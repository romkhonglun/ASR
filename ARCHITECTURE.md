# TDT-VI Architecture

Token-and-Duration Transducer for Vietnamese Automatic Speech Recognition.

## Overview

TDT-VI is a custom ASR model that combines:
- **Frozen Encoder**: NVIDIA Parakeet FastConformer CTC (0.6B params, pretrained on Vietnamese)
- **Trainable Decoder**: TDT (Token-and-Duration Transducer) decoder built with NeMo modules

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TDT-VI Model                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Audio (16kHz)                                                      │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────┐                   │
│  │        Parakeet Encoder (FROZEN)            │                   │
│  │  ┌───────────────────────────────────────┐  │                   │
│  │  │  Preprocessor (Mel Spectrogram)       │  │                   │
│  │  │    ↓                                   │  │                   │
│  │  │  8x Depthwise-Separable Downsampling  │  │                   │
│  │  │    ↓                                   │  │                   │
│  │  │  FastConformer Encoder Blocks         │  │                   │
│  │  │    (~600M params, all frozen)         │  │                   │
│  │  └───────────────────────────────────────┘  │                   │
│  │  Output: Hidden States [T, encoder_dim]     │                   │
│  └─────────────────────────────────────────────┘                   │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────┐                   │
│  │         TDT Decoder (TRAINABLE)             │                   │
│  │  ┌───────────────────────────────────────┐  │                   │
│  │  │  Predictor (LSTM-based)               │  │                   │
│  │  │    - Token Embedding                  │  │                   │
│  │  │    - LSTM Layers                      │  │                   │
│  │  │    Output: Context [U, predictor_dim] │  │                   │
│  │  └───────────────────────────────────────┘  │                   │
│  │              │                               │                   │
│  │              ▼                               │                   │
│  │  ┌───────────────────────────────────────┐  │                   │
│  │  │  Joint Network (2-headed)             │  │                   │
│  │  │    ┌─────────────────────────────┐    │  │                   │
│  │  │    │  Encoder + Decoder concat   │    │  │                   │
│  │  │    │    ↓                         │    │  │                   │
│  │  │    │  Shared Hidden Layers       │    │  │                   │
│  │  │    │    ↓                         │    │  │                   │
│  │  │    ├────────────┬──────────────┐ │    │  │                   │
│  │  │    │ Token Head │ Duration Head│ │    │  │                   │
│  │  │    │ P(token)   │ P(duration)  │ │    │  │                   │
│  │  │    └────────────┴──────────────┘ │    │  │                   │
│  │  └─────────────────────────────────┘    │  │                   │
│  └─────────────────────────────────────────┘  │                   │
│                                                │                   │
└────────────────────────────────────────────────┴───────────────────┘
```

## Components

### 1. Parakeet Encoder (`models/components/encoder.py`)

- **Source**: `nvidia/parakeet-ctc-0.6b-Vietnamese` (.nemo file)
- **Architecture**: FastConformer with 8x downsampling
- **Parameters**: ~600M (all frozen)
- **Input**: Raw audio [batch, time] at 16kHz
- **Output**: Hidden states [batch, frames, encoder_dim] at ~3.125ms/frame
- **Key feature**: Frozen during training, acts as feature extractor

### 2. TDT Predictor (`models/components/predictor.py`)

- **Type**: Autoregressive RNN decoder (LSTM/GRU)
- **Parameters**: Trainable (~1-10M depending on config)
- **Input**: Token IDs [batch, seq_len]
- **Output**: Context vectors [batch, seq_len, predictor_dim]
- **Key feature**: Provides linguistic context for joint prediction

### 3. TDT Joint (`models/components/joint.py`)

- **Type**: Two-headed neural network
- **Heads**:
  - **Token head**: P(token | t, u) over vocabulary
  - **Duration head**: P(duration | t, u) over {0, 1, ..., max_duration}
- **Key innovation**: Decouples "what to say" from "how long to say it"

### 4. TDT-VI Model (`models/tdt_vi.py`)

Combines all components into a complete ASR model with:
- Forward pass for training (teacher forcing)
- Autoregressive decoding for inference (frame-skipping)
- Optional CTC warmup head

## Training Strategy

### Stage 1: CTC Warmup (Optional)
- Train predictor + joint with CTC-like objective
- Helps initialize decoder before full TDT training
- Uses `ctc_warmup.enabled=true` in config

### Stage 2: Full TDT Training
- Train with TDT loss (token + duration)
- Encoder remains frozen
- Uses forward-backward algorithm over token-duration lattice

### Current Simplification
The current implementation uses a **simplified TDT loss** (cross-entropy proxy):
- Token loss: Cross-entropy on token logits
- Duration loss: Cross-entropy on duration logits (target=1)

For production, replace with proper TDT forward-backward loss from NeMo.

## Inference (Frame-Skipping Decoding)

TDT enables faster decoding through frame-skipping:

```
t=0, u=0: Predict token + duration
    → If token != blank: output token, t += duration
    → If token == blank: t += max(1, duration)
Repeat until t >= audio_length
```

This is 2-3x faster than standard RNN-T decoding.

## Configuration

All configs are in `conf/` directory:

| File | Purpose |
|------|---------|
| `conf/config.yaml` | Main config (includes all others) |
| `conf/model/tdt-vi.yaml` | TDT-VI model settings |
| `conf/model/parakeet-encoder.yaml` | Encoder config (freeze, projection) |
| `conf/decoder/tdt.yaml` | TDT decoder architecture |
| `conf/trainer/lightning.yaml` | Lightning trainer settings |
| `conf/dataset/vlsp2020.yaml` | Dataset configuration |
| `conf/logging/wandb.yaml` | W&B logging |

### CLI Overrides

```bash
# Change learning rate
python -m asr.training.train optimizer.lr=5e-5

# Increase epochs
python -m asr.training.train trainer.max_epochs=100

# Disable W&B
python -m asr.training.train ~logging/wandb

# Resume from checkpoint
python -m asr.training.train model.checkpoint_path=path/to/last.ckpt

# Change decoder size
python -m asr.training.train model.decoder.predictor.hidden_dim=512
```

## File Structure

```
src/asr/
├── models/
│   ├── __init__.py
│   ├── tdt_vi.py                    # Main model
│   └── components/
│       ├── __init__.py
│       ├── encoder.py               # Parakeet encoder wrapper
│       ├── predictor.py             # TDT predictor (RNN)
│       └── joint.py                 # TDT joint network
├── training/
│   ├── __init__.py
│   ├── train.py                     # Entry point (@hydra.main)
│   ├── lightning_module.py          # LightningModule
│   └── datamodule.py                # LightningDataModule
└── api/                             # (future)

conf/
├── config.yaml
├── model/
│   ├── tdt-vi.yaml
│   └── parakeet-encoder.yaml
├── decoder/
│   └── tdt.yaml
├── trainer/
│   └── lightning.yaml
└── ...
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Core framework |
| `lightning` | Training orchestration |
| `nemo-toolkit[asr]` | Parakeet model loading, TDT modules |
| `hydra-core` | Configuration management |
| `wandb` | Experiment tracking |
| `datasets` | Dataset loading |
| `librosa` | Audio preprocessing |
| `jiwer` | WER/CER metrics |

## Future Improvements

- [ ] Implement proper TDT forward-backward loss
- [ ] Add LoRA/PEFT for encoder unfreezing
- [ ] Data augmentation (noise, speed perturbation)
- [ ] ONNX export for deployment
- [ ] Streaming inference support
- [ ] n-gram language model fusion
