"""
Lightning Module for TDT-VI Training

Wraps the TDT-VI model with PyTorch Lightning for training, validation,
optimizer/scheduler setup, metrics logging, and W&B integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from typing import Optional
from omegaconf import DictConfig
import wandb

from ..models import TDTVIModel


class TDTVILightningModule(pl.LightningModule):
    """
    Lightning Module for TDT-VI ASR training.

    Handles:
        - Training/validation steps
        - TDT loss computation (token + duration)
        - CTC warmup loss (optional)
        - WER/CER metrics
        - Optimizer & scheduler configuration
        - W&B logging
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Initialize model
        self.model = TDTVIModel(config)

        # Loss weights
        training_strategy = config.model.get("training_strategy", {})
        self.ctc_loss_weight = training_strategy.get("ctc_warmup", {}).get("loss_weight", 0.0)
        self.tdt_loss_weight = training_strategy.get("tdt", {}).get("loss_weight", 1.0)

        # CTC loss function (for warmup)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Metrics tracking
        self.train_losses = []
        self.val_wer = float("inf")
        self.val_cer = float("inf")

        # W&B logging
        self.wandb_config = config.get("logging", {}).get("wandb", {})
        self.log_interval = self.wandb_config.get("log_interval", 50)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(audio, audio_lengths, tokens, token_lengths)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Dict with 'audio', 'audio_lengths', 'tokens', 'token_lengths'
            batch_idx: Batch index

        Returns:
            Total loss
        """
        audio = batch["audio"]
        audio_lengths = batch["audio_lengths"]
        tokens = batch["tokens"]
        token_lengths = batch["token_lengths"]

        # Forward pass
        outputs = self.model(audio, audio_lengths, tokens, token_lengths)

        # Compute losses
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}

        # TDT loss (token + duration)
        if self.tdt_loss_weight > 0:
            tdt_loss = self._compute_tdt_loss(
                outputs["token_logits"],
                outputs["duration_logits"],
                tokens,
                token_lengths,
                outputs["encoder_lengths"]
            )
            total_loss = total_loss + tdt_loss * self.tdt_loss_weight
            loss_dict["tdt_loss"] = tdt_loss.item()

        # CTC warmup loss (optional)
        if self.ctc_loss_weight > 0 and "ctc_logits" in outputs:
            ctc_loss = self._compute_ctc_loss(
                outputs["ctc_logits"],
                tokens,
                outputs["encoder_lengths"],
                token_lengths
            )
            total_loss = total_loss + ctc_loss * self.ctc_loss_weight
            loss_dict["ctc_loss"] = ctc_loss.item()

        # Log losses
        self.log("train/total_loss", total_loss, prog_bar=True, sync_dist=True)
        for key, val in loss_dict.items():
            self.log(f"train/{key}", val, prog_bar=False, sync_dist=True)

        # Log learning rate
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], prog_bar=False)

        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """
        Validation step.

        Computes loss and collects predictions for WER/CER calculation.
        """
        audio = batch["audio"]
        audio_lengths = batch["audio_lengths"]
        tokens = batch["tokens"]
        token_lengths = batch["token_lengths"]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(audio, audio_lengths, tokens, token_lengths)

        # Compute validation loss
        total_loss = torch.tensor(0.0, device=self.device)

        if self.tdt_loss_weight > 0:
            tdt_loss = self._compute_tdt_loss(
                outputs["token_logits"],
                outputs["duration_logits"],
                tokens,
                token_lengths,
                outputs["encoder_lengths"]
            )
            total_loss = total_loss + tdt_loss * self.tdt_loss_weight

        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)

        # Compute WER/CER (on last batch only to save time)
        if batch_idx == 0:
            wer, cer = self._compute_wer_cer(outputs, tokens)
            self.log("val/wer", wer, prog_bar=True, sync_dist=True)
            self.log("val/cer", cer, prog_bar=True, sync_dist=True)

            # Update tracking
            self.val_wer = wer
            self.val_cer = cer

    def on_validation_epoch_end(self) -> None:
        """Log epoch-level metrics."""
        self.log("val/wer_epoch", self.val_wer, prog_bar=True, sync_dist=True)
        self.log("val/cer_epoch", self.val_cer, prog_bar=True, sync_dist=True)

    def _compute_tdt_loss(
        self,
        token_logits: torch.Tensor,
        duration_logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        input_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TDT loss.

        NOTE: This is a simplified version using cross-entropy as proxy.
        The full TDT loss requires forward-backward algorithm over the
        token-duration lattice (similar to RNNT loss but with duration axis).

        For production, this should be replaced with proper TDT loss from
        NeMo or custom CUDA implementation.

        Simplified approach:
            - Token loss: Cross-entropy on token logits (teacher forcing)
            - Duration loss: Cross-entropy on duration logits (assuming duration=1 as target)
        """
        batch_size, time_len, seq_len, vocab_size = token_logits.shape

        # Token loss: average over time and sequence dimensions
        token_logits_flat = token_logits.view(-1, vocab_size)
        targets_expanded = targets.unsqueeze(1).expand(-1, time_len, -1).reshape(-1)
        token_loss = F.cross_entropy(
            token_logits_flat,
            targets_expanded,
            ignore_index=0,  # Ignore blank/padding
            reduction="mean"
        )

        # Duration loss: simple cross-entropy (target duration = 1 for now)
        max_dur = duration_logits.size(-1)
        duration_targets = torch.ones_like(duration_logits[..., 0], dtype=torch.long)
        duration_logits_flat = duration_logits.view(-1, max_dur)
        duration_loss = F.cross_entropy(
            duration_logits_flat,
            duration_targets.view(-1),
            reduction="mean"
        )

        # Combined loss
        total_loss = token_loss + 0.1 * duration_loss

        return total_loss

    def _compute_ctc_loss(
        self,
        ctc_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute CTC loss."""
        # Transpose logits: [batch, time, vocab] -> [time, batch, vocab]
        ctc_logits = ctc_logits.transpose(0, 1)

        # Compute CTC loss
        loss = self.ctc_loss(
            ctc_logits,  # [time, batch, vocab]
            targets,  # [batch, max_target_len]
            input_lengths,  # [batch]
            target_lengths  # [batch]
        )

        return loss

    def _compute_wer_cer(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> tuple[float, float]:
        """
        Compute WER and CER metrics.

        Simplified version - in production, should use proper tokenization
        and jiwer library.
        """
        try:
            from jiwer import wer, cer

            # Get predictions (argmax over token logits at last time step)
            token_logits = outputs["token_logits"]
            pred_tokens = token_logits[:, -1, :, :].argmax(dim=-1)  # [batch, seq_len]

            # Decode to strings (placeholder - needs proper tokenizer)
            pred_texts = [self._decode_tokens(pred_tokens[i]) for i in range(pred_tokens.size(0))]
            target_texts = [self._decode_tokens(targets[i]) for i in range(targets.size(0))]

            # Filter empty strings
            valid_pairs = [(p, t) for p, t in zip(pred_texts, target_texts) if t.strip()]
            if not valid_pairs:
                return 1.0, 1.0

            pred_valid, target_valid = zip(*valid_pairs)

            wer_score = wer(list(target_valid), list(pred_valid))
            cer_score = cer(list(target_valid), list(pred_valid))

            return wer_score, cer_score
        except Exception:
            return 1.0, 1.0

    def _decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text (placeholder)."""
        return ""

    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler."""
        opt_config = self.config.get("optimizer", {})
        sched_config = self.config.get("trainer", {}).get("scheduler", {})

        # Get trainable parameters (exclude frozen encoder)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Setup optimizer
        opt_name = opt_config.get("name", "adamw").lower()
        lr = opt_config.get("lr", 1e-4)
        weight_decay = opt_config.get("weight_decay", 0.01)
        betas = opt_config.get("betas", [0.9, 0.98])
        eps = opt_config.get("eps", 1e-8)

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(betas),
                eps=eps
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
                betas=tuple(betas),
                eps=eps
            )
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        # Setup scheduler
        sched_name = sched_config.get("name", "cosine_warmup")
        warmup_steps = sched_config.get("warmup_steps", 5000)

        if "cosine" in sched_name.lower():
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=warmup_steps * 10,
                    eta_min=1e-7
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=10),
                "interval": "epoch",
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_epoch_end(self) -> None:
        """Log epoch summary."""
        avg_loss = torch.tensor(self.train_losses).mean() if self.train_losses else 0
        self.log("train/epoch_loss", avg_loss, sync_dist=True)
        self.train_losses.clear()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Track batch loss for epoch average."""
        if "loss" in outputs:
            self.train_losses.append(outputs["loss"].item())
