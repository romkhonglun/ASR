"""
Tests for Lightning Module
"""

import pytest
import torch
from omegaconf import OmegaConf

from asr.training.lightning_module import TDTVILightningModule


@pytest.mark.skip(reason="Requires NeMo model loading - slow test")
class TestTDTVILightningModule:
    """Tests for TDTVILightningModule (requires NeMo)."""

    @pytest.fixture
    def lightning_module(self, default_config):
        return TDTVILightningModule(default_config)

    def test_init(self, lightning_module):
        """Test initialization."""
        assert lightning_module.model is not None
        assert lightning_module.config is not None

    def test_training_step(self, lightning_module, mock_batch):
        """Test training step."""
        lightning_module.train()
        loss = lightning_module.training_step(mock_batch, 0)
        assert loss.dim() == 0  # Scalar tensor

    def test_validation_step(self, lightning_module, mock_batch):
        """Test validation step."""
        lightning_module.eval()
        lightning_module.validation_step(mock_batch, 0)

    def test_configure_optimizers(self, lightning_module):
        """Test optimizer configuration."""
        opt_config = lightning_module.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config


class TestTDTVILightningModuleCPU:
    """CPU-only tests for Lightning Module (no NeMo dependency)."""

    @pytest.fixture
    def cpu_config(self):
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
                        "vocab_size": 100,
                        "embedding_dim": 32,
                        "hidden_dim": 32,
                        "num_layers": 1,
                        "dropout": 0.1,
                        "rnn_type": "lstm",
                        "blank_idx": 0,
                        "pad_idx": 0,
                    },
                    "duration": {"max_duration": 3, "embedding_dim": 8},
                    "joint": {
                        "hidden_dim": 32,
                        "activation": "relu",
                        "dropout": 0.1,
                        "token_head": {"hidden_dim": 32},
                        "duration_head": {"hidden_dim": 16},
                    },
                },
                "tokenizer": {"type": "sentencepiece", "model_path": None, "vocab_size": 100},
                "training_strategy": {
                    "ctc_warmup": {"enabled": False, "epochs": 5, "loss_weight": 1.0},
                    "tdt": {"enabled": True, "loss_weight": 1.0},
                },
                "inference": {"beam_size": 1, "max_symbols_per_step": 2, "temperature": 1.0},
            },
            "encoder": {
                "pretrained_model_name": "nvidia/parakeet-ctc-0.6b-Vietnamese",
                "nemo_path": None,
                "freeze": True,
            },
            "decoder": {
                "predictor": {
                    "vocab_size": 100,
                    "embedding_dim": 32,
                    "hidden_dim": 32,
                    "num_layers": 1,
                },
                "joint": {
                    "hidden_dim": 32,
                    "token_head": {"hidden_dim": 32},
                    "duration_head": {"hidden_dim": 16},
                },
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
            },
            "logging": {
                "wandb": {"enabled": False, "project": "test", "log_interval": 10}
            },
        }
        return OmegaConf.create(config_dict)

    def test_config_structure(self, cpu_config):
        """Test config structure."""
        assert cpu_config.model.decoder.predictor.vocab_size == 100
        assert cpu_config.optimizer.name == "adamw"
