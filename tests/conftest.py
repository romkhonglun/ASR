"""
Test fixtures and configuration
"""

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config():
    """Default configuration for testing."""
    config_dict = {
        "model": {
            "name": "tdt-vi-test",
            "encoder": {
                "pretrained_model_name": "nvidia/parakeet-ctc-0.6b-Vietnamese",
                "nemo_path": None,
                "freeze": True,
                "freeze_exclude": [],
                "projection": {"enabled": False, "out_dim": None},
                "sample_rate": 16000,
                "n_feats": 80,
            },
            "decoder": {
                "type": "tdt",
                "predictor": {
                    "vocab_size": 1024,
                    "embedding_dim": 128,
                    "hidden_dim": 128,
                    "num_layers": 1,
                    "dropout": 0.1,
                    "rnn_type": "lstm",
                    "blank_idx": 0,
                    "pad_idx": 0,
                },
                "duration": {"max_duration": 5, "embedding_dim": 16},
                "joint": {
                    "hidden_dim": 128,
                    "activation": "relu",
                    "dropout": 0.1,
                    "token_head": {"hidden_dim": 128},
                    "duration_head": {"hidden_dim": 32},
                },
            },
            "tokenizer": {"type": "sentencepiece", "model_path": None, "vocab_size": 1024},
            "training_strategy": {
                "ctc_warmup": {"enabled": False, "epochs": 5, "loss_weight": 1.0},
                "tdt": {"enabled": True, "loss_weight": 1.0},
            },
            "inference": {"beam_size": 1, "max_symbols_per_step": 2, "temperature": 1.0},
            "checkpoint_path": None,
            "pretrained_path": None,
        },
        "encoder": {
            "pretrained_model_name": "nvidia/parakeet-ctc-0.6b-Vietnamese",
            "nemo_path": None,
            "freeze": True,
            "freeze_exclude": [],
            "projection": {"enabled": False, "out_dim": None},
            "sample_rate": 16000,
            "n_feats": 80,
        },
        "decoder": {
            "type": "tdt",
            "predictor": {
                "vocab_size": 1024,
                "embedding_dim": 128,
                "hidden_dim": 128,
                "num_layers": 1,
                "dropout": 0.1,
                "rnn_type": "lstm",
                "blank_idx": 0,
                "pad_idx": 0,
            },
            "duration": {"max_duration": 5, "embedding_dim": 16},
            "joint": {
                "hidden_dim": 128,
                "activation": "relu",
                "dropout": 0.1,
                "token_head": {"hidden_dim": 128},
                "duration_head": {"hidden_dim": 32},
            },
        },
        "dataset": {
            "dataset_name": "doof-ferb/vlsp2020_vinai_100h",
            "train_split": "train",
            "eval_split": "test",
            "sampling_rate": 16000,
            "audio_column_name": "audio",
            "text_column_name": "sentence",
        },
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32",
            "max_epochs": 1,
            "scheduler": {"name": "cosine_warmup", "warmup_steps": 100},
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
        },
        "logging": {
            "wandb": {
                "enabled": False,
                "project": "asr-vietnamese-test",
                "log_interval": 10,
            }
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def dummy_audio(device):
    """Generate dummy audio (1 second at 16kHz)."""
    return torch.randn(1, 16000, device=device)


@pytest.fixture
def dummy_audio_batch(device):
    """Generate dummy audio batch (batch_size=2, 1 second at 16kHz)."""
    return torch.randn(2, 16000, device=device)


@pytest.fixture
def dummy_tokens():
    """Generate dummy token sequence."""
    return torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]], dtype=torch.long)


@pytest.fixture
def mock_batch(dummy_audio_batch, dummy_tokens, device):
    """Generate a complete mock batch."""
    batch_size = dummy_audio_batch.size(0)
    return {
        "audio": dummy_audio_batch,
        "audio_lengths": torch.tensor([16000] * batch_size, device=device),
        "tokens": dummy_tokens,
        "token_lengths": torch.tensor([5, 3], dtype=torch.long),
    }
