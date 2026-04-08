# TDT-VI: Token-and-Duration Transducer for Vietnamese ASR

Hệ thống nhận dạng giọng nói tiếng Việt sử dụng kiến trúc **TDT (Token-and-Duration Transducer)** với:
- **Encoder**: NVIDIA Parakeet FastConformer CTC 0.6B (frozen, pretrained trên tiếng Việt)
- **Decoder**: TDT decoder (trainable) - dự đoán đồng thời token + duration
- **Framework**: PyTorch Lightning + Hydra + W&B

> **Chi tiết kiến trúc**: [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## Quick Start (Server RTX 3090 + CUDA 12.9)

### 1. Cài đặt môi trường

```bash
# Clone repo
git clone <repo-url>
cd ASR

# Cài đặt toàn bộ dependencies (PyTorch CUDA 12.9 + NeMo + tools)
pixi install

# Kiểm tra CUDA
pixi run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### 2. Chạy training

```bash
# Default training
pixi run train

# Training với W&B logging
WANDB_API_KEY=your_key pixi run train

# Override config từ CLI
pixi run train trainer.max_epochs=100 optimizer.lr=5e-5
pixi run train model.decoder.predictor.hidden_dim=512

# Resume từ checkpoint
pixi run train model.checkpoint_path=outputs/training/checkpoints/last.ckpt
```

---

## Yêu cầu hệ thống

| Component | Yêu cầu |
|-----------|---------|
| GPU | NVIDIA RTX 3090 (24GB VRAM) hoặc tương đương |
| CUDA | 12.9 |
| RAM | 32GB+ |
| Storage | 50GB+ (model + dataset + checkpoints) |
| Python | 3.11 - 3.12 |

---

## Cấu trúc project

```
ASR/
├── conf/                          # Hydra configuration
│   ├── config.yaml                # Main config
│   ├── model/
│   │   ├── tdt-vi.yaml            # TDT-VI model settings
│   │   └── parakeet-encoder.yaml  # Encoder config
│   ├── decoder/
│   │   └── tdt.yaml               # TDT decoder architecture
│   ├── trainer/
│   │   └── lightning.yaml         # Lightning trainer
│   ├── dataset/
│   │   └── vlsp2020.yaml          # Dataset config
│   ├── logging/
│   │   └── wandb.yaml             # W&B logging
│   └── cache/
│       └── redis.yaml             # Redis caching
├── src/asr/
│   ├── models/
│   │   ├── tdt_vi.py              # Main TDT-VI model
│   │   └── components/
│   │       ├── encoder.py         # Parakeet encoder wrapper
│   │       ├── predictor.py       # TDT predictor (RNN)
│   │       └── joint.py           # TDT joint network
│   ├── training/
│   │   ├── train.py               # Entry point (@hydra.main)
│   │   ├── lightning_module.py    # LightningModule
│   │   └── datamodule.py          # LightningDataModule
│   ├── api/                       # (future)
│   └── utils/                     # (future)
├── tests/
│   ├── conftest.py
│   ├── test_tdt_components.py
│   └── test_lightning_module.py
├── ARCHITECTURE.md                # Architecture documentation
├── pyproject.toml
└── .pre-commit-config.yaml
```

---

## Configuration

### Hydra config overrides

Tất cả config có thể override từ CLI:

```bash
# Training params
pixi run train trainer.max_epochs=100
pixi run train trainer.accumulate_grad_batches=8
pixi run train optimizer.lr=1e-4
pixi run train optimizer.weight_decay=0.01

# Model architecture
pixi run train model.decoder.predictor.hidden_dim=512
pixi run train model.decoder.predictor.num_layers=2
pixi run train model.decoder.joint.hidden_dim=512

# Dataset
pixi run train dataset.max_train_samples=10000
pixi run train dataset.max_eval_samples=1000

# Logging
pixi run train ~logging/wandb                    # Disable W&B
pixi run train logging.wandb.project=my-project  # Change project name

# Checkpoints
pixi run train model.checkpoint_path=outputs/training/checkpoints/last.ckpt
```

### Config files chính

| File | Mô tả |
|------|-------|
| `conf/trainer/lightning.yaml` | GPU, precision, epochs, gradient clipping, early stopping |
| `conf/decoder/tdt.yaml` | Predictor (LSTM), Joint network, loss config |
| `conf/model/parakeet-encoder.yaml` | Encoder freeze, projection, feature extraction |
| `conf/dataset/vlsp2020.yaml` | Dataset path, splits, sampling rate |

---

## Training trên Server

### Setup lần đầu

```bash
# 1. SSH vào server
ssh user@your-server

# 2. Cài đặt pixi (nếu chưa có)
curl -fsSL https://pixi.sh/install.sh | bash

# 3. Clone repo
git clone <repo-url>
cd ASR

# 4. Cài đặt environment (~10-15 phút, tải PyTorch + NeMo)
pixi install

# 5. Login W&B (optional)
wandb login
```

### Chạy training

```bash
# Cơ bản
pixi run train

# Với W&B logging để theo dõi từ xa
pixi run train logging.wandb.enabled=true

# Multi-GPU (nếu có nhiều GPU)
pixi run train trainer.devices=2 trainer.strategy=ddp

# Precision thấp hơn để nhanh hơn (RTX 3090 hỗ trợ FP16 tốt)
pixi run train trainer.precision="16-mixed"

# Gradient accumulation lớn hơn nếu VRAM ít
pixi run train trainer.accumulate_grad_batches=8
```

### Monitoring

- **W&B Dashboard**: Mở `https://wandb.ai/your-project` để xem loss, WER, CER real-time
- **GPU Usage**: `watch -n 1 nvidia-smi`
- **Logs**: Xem trong `outputs/training/logs/`

### Kết quả training

Sau khi train xong:
```
outputs/training/
├── checkpoints/
│   ├── epoch=10-val_wer=0.1234.ckpt
│   ├── epoch=15-val_wer=0.1123.ckpt   ← best model
│   └── last.ckpt
├── logs/
└── model_config.json
```

---

## Development

### Chạy tests

```bash
pixi run test
```

### Linting & Formatting

```bash
pixi run lint      # Ruff check + fix
pixi run format    # Ruff format
pixi run typecheck # Pyrefly type check
```

### Pre-commit hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Dataset

Project sử dụng dataset **VLSP 2020** (~100 hours tiếng Việt):
- HuggingFace: `doof-ferb/vlsp2020_vinai_100h`
- Tự động download khi chạy training lần đầu
- Config: `conf/dataset/vlsp2020.yaml`

---

## Kiến trúc TDT-VI

```
Audio (16kHz)
    │
    ▼
┌─────────────────────────────┐
│  Parakeet Encoder (FROZEN)  │
│  FastConformer, ~600M params│
│  Output: Hidden States      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    TDT Decoder (TRAIN)      │
│  ┌───────────────────────┐  │
│  │  Predictor (LSTM)     │  │
│  │  Output: Context      │  │
│  └──────────┬────────────┘  │
│             │                │
│             ▼                │
│  ┌───────────────────────┐  │
│  │  Joint (2-headed)     │  │
│  │  ├─ Token Head        │  │
│  │  └─ Duration Head     │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
               │
               ▼
        Tokens + Durations
```

**Ưu điểm:**
- Encoder đã pre-trained tốt → không cần train lại
- TDT decoder nhẹ → train nhanh, ít VRAM
- Frame-skipping inference → nhanh hơn 2-3x so với RNN-T

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Giảm batch size, tăng gradient accumulation
pixi run train trainer.batch_size=8 trainer.accumulate_grad_batches=8

# Dùng precision thấp hơn
pixi run train trainer.precision="16-mixed"
```

### NeMo installation lỗi

```bash
# Đảm bảo CUDA 12.9 và PyTorch đúng version
pixi run python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Cài lại
pixi reinstall
```

### Dataset load chậm

```bash
# Giới hạn samples cho debugging
pixi run train dataset.max_train_samples=1000 dataset.max_eval_samples=100
```

---

## License

Internal project - For internal use only.
