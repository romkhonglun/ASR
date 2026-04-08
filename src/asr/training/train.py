"""
Training entry point for TDT-VI ASR

Usage:
    # Default training
    python -m asr.training.train

    # Override config from CLI
    python -m asr.training.train trainer.max_epochs=100
    python -m asr.training.train model.tokenizer.vocab_size=2048
    python -m asr.training.train optimizer.lr=5e-5
    python -m asr.training.train ~logging/wandb  # Disable W&B

    # Resume from checkpoint
    python -m asr.training.train model.checkpoint_path=outputs/training/checkpoints/last.ckpt
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from .lightning_module import TDTVILightningModule
from .datamodule import TDTDataModule


@hydra.main(
    config_path="../../../conf",
    config_name="config",
    version_base="1.3"
)
def main(config: DictConfig) -> None:
    """
    Main training function with Hydra configuration.

    Args:
        config: Full configuration from Hydra
    """
    # Print config
    print("\n" + "=" * 80)
    print("TDT-VI Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80 + "\n")

    # Set seed for reproducibility
    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # Initialize Lightning Module
    lightning_module = TDTVILightningModule(config)

    # Print model stats
    model_info = lightning_module.model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Total: {model_info['total']:,}")
    print(f"  Trainable: {model_info['trainable']:,}")
    print(f"  Frozen: {model_info['frozen']:,}")
    print(f"  Encoder: {model_info['encoder']:,}")
    print(f"  Predictor: {model_info['predictor']:,}")
    print(f"  Joint: {model_info['joint']:,}\n")

    # Initialize DataModule
    datamodule = TDTDataModule(config)

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    ckpt_config = config.trainer.get("model_checkpoint", {})
    if ckpt_config.get("enabled", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(config.model.get("output_dir", "outputs/training"), "checkpoints"),
            monitor=ckpt_config.get("monitor", "val/wer"),
            mode=ckpt_config.get("mode", "min"),
            save_top_k=ckpt_config.get("save_top_k", 3),
            save_last=True,
            every_n_epochs=ckpt_config.get("every_n_epochs", 1),
            filename=ckpt_config.get("filename", "{epoch}-{val_wer:.4f}"),
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    # Early stopping
    es_config = config.trainer.get("early_stopping", {})
    if es_config.get("enabled", True):
        early_stopping = EarlyStopping(
            monitor=es_config.get("monitor", "val/wer"),
            patience=es_config.get("patience", 10),
            mode=es_config.get("mode", "min"),
            verbose=True,
        )
        callbacks.append(early_stopping)

    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Setup W&B logger
    wandb_config = config.get("logging", {}).get("wandb", {})
    wandb_logger = None
    if wandb_config.get("enabled", True):
        wandb_logger = WandbLogger(
            project=wandb_config.get("project", "asr-vietnamese"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("run_name"),
            group=wandb_config.get("group"),
            save_code=wandb_config.get("save_code", True),
            offline=wandb_config.get("offline", False),
            config=OmegaConf.to_container(config, resolve=True),
        )

    # Initialize Lightning Trainer
    trainer_config = config.get("trainer", {})
    trainer = pl.Trainer(
        accelerator=trainer_config.get("accelerator", "gpu"),
        devices=trainer_config.get("devices", 1),
        strategy=trainer_config.get("strategy", "auto"),
        precision=trainer_config.get("precision", "16-mixed"),
        max_epochs=trainer_config.get("max_epochs", 50),
        min_epochs=trainer_config.get("min_epochs", 1),
        gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 4),
        val_check_interval=trainer_config.get("val_check_interval", 0.5),
        log_every_n_steps=trainer_config.get("log_every_n_steps", 50),
        callbacks=callbacks,
        logger=wandb_logger,
    )

    # Load checkpoint if specified
    ckpt_path = config.model.get("checkpoint_path")

    # Train
    print("\nStarting training...")
    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    # Save final model
    output_dir = config.model.get("output_dir", "outputs/training")
    if trainer.is_global_zero:
        print(f"\nTraining completed. Saving model to {output_dir}")
        # Save model config for inference later
        model_config = lightning_module.model.get_config()
        import json
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
