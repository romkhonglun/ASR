"""
Tests for TDT components (Predictor, Joint)
"""

import pytest
import torch
from omegaconf import OmegaConf

from asr.models.components.predictor import TDTPredictor
from asr.models.components.joint import TDTJoint


class TestTDTPredictor:
    """Tests for TDTPredictor."""

    @pytest.fixture
    def predictor_config(self):
        return OmegaConf.create({
            "vocab_size": 1024,
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 1,
            "dropout": 0.1,
            "rnn_type": "lstm",
            "blank_idx": 0,
            "pad_idx": 0,
        })

    @pytest.fixture
    def predictor(self, predictor_config):
        return TDTPredictor(predictor_config)

    def test_predictor_init(self, predictor):
        """Test predictor initialization."""
        assert predictor.vocab_size == 1024
        assert predictor.hidden_dim == 128
        assert predictor.output_dim == 128
        assert predictor.rnn_type == "lstm"

    def test_forward(self, predictor, device):
        """Test forward pass."""
        batch_size = 2
        seq_len = 10
        tokens = torch.randint(0, 1024, (batch_size, seq_len), device=device)

        output, hidden = predictor(tokens)

        assert output.shape == (batch_size, seq_len, predictor.hidden_dim)
        assert hidden is not None

    def test_forward_step(self, predictor, device):
        """Test single-step forward."""
        batch_size = 2
        token = torch.randint(0, 1024, (batch_size, 1), device=device)

        output, hidden = predictor.forward_step(token)

        assert output.shape == (batch_size, 1, predictor.hidden_dim)

    def test_initial_state(self, predictor, device):
        """Test initial state creation."""
        batch_size = 4
        h0, c0 = predictor.get_initial_state(batch_size, device)

        assert h0.shape == (predictor.num_layers, batch_size, predictor.hidden_dim)
        assert c0.shape == (predictor.num_layers, batch_size, predictor.hidden_dim)

    def test_gru_predictor(self):
        """Test GRU-based predictor."""
        config = OmegaConf.create({
            "vocab_size": 512,
            "embedding_dim": 64,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "rnn_type": "gru",
            "blank_idx": 0,
            "pad_idx": 0,
        })
        predictor = TDTPredictor(config)
        assert predictor.rnn_type == "gru"


class TestTDTJoint:
    """Tests for TDTJoint."""

    @pytest.fixture
    def joint_config(self):
        return OmegaConf.create({
            "hidden_dim": 128,
            "activation": "relu",
            "dropout": 0.1,
            "token_head": {"hidden_dim": 128},
            "duration_head": {"hidden_dim": 32},
        })

    @pytest.fixture
    def joint(self, joint_config):
        joint = TDTJoint(joint_config, encoder_dim=128, predictor_dim=128)
        joint.setup_outputs(vocab_size=1024, max_duration=5)
        return joint

    def test_joint_init(self, joint):
        """Test joint initialization."""
        assert joint.encoder_dim == 128
        assert joint.predictor_dim == 128
        assert joint.max_duration == 5

    def test_forward(self, joint, device):
        """Test forward pass."""
        batch_size = 2
        time_len = 50
        seq_len = 10

        encoder_hidden = torch.randn(batch_size, time_len, 128, device=device)
        predictor_hidden = torch.randn(batch_size, seq_len, 128, device=device)

        token_logits, duration_logits = joint(encoder_hidden, predictor_hidden)

        assert token_logits.shape == (batch_size, time_len, seq_len, 1024)
        assert duration_logits.shape == (batch_size, time_len, seq_len, 6)  # max_duration + 1

    def test_forward_step(self, joint, device):
        """Test single-step forward."""
        batch_size = 2

        encoder_frame = torch.randn(batch_size, 128, device=device)
        predictor_frame = torch.randn(batch_size, 128, device=device)

        token_probs, duration_probs = joint.forward_step(encoder_frame, predictor_frame)

        assert token_probs.shape == (batch_size, 1024)
        assert duration_probs.shape == (batch_size, 6)

        # Check probabilities sum to 1
        assert torch.allclose(token_probs.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5)
        assert torch.allclose(duration_probs.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5)

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "gelu", "silu"]:
            config = OmegaConf.create({
                "hidden_dim": 64,
                "activation": activation,
                "dropout": 0.1,
                "token_head": {"hidden_dim": 64},
                "duration_head": {"hidden_dim": 16},
            })
            joint = TDTJoint(config, encoder_dim=64, predictor_dim=64)
            joint.setup_outputs(vocab_size=100, max_duration=3)
            # Should not raise
