"""
Lightning DataModule for TDT-VI Training

Handles dataset loading, preprocessing, tokenization, and DataLoader creation
for the TDT-VI training pipeline.
"""

import torch
import lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Callable
from omegaconf import DictConfig
from datasets import Dataset as HFDataset


class TDTDataModule(pl.LightningDataModule):
    """
    DataModule for TDT-VI ASR training.

    Responsibilities:
        - Load VLSP2020 dataset (or custom dataset)
        - Audio preprocessing (resample, normalize)
        - Tokenization (convert text to token IDs)
        - Create DataLoaders with proper collation

    Args:
        config: Full Hydra configuration
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Dataset config
        ds_config = config.get("dataset", {})
        self.dataset_name = ds_config.get("dataset_name", "doof-ferb/vlsp2020_vinai_100h")
        self.train_split = ds_config.get("train_split", "train")
        self.eval_split = ds_config.get("eval_split", "test")
        self.max_train_samples = ds_config.get("max_train_samples")
        self.max_eval_samples = ds_config.get("max_eval_samples")
        self.sampling_rate = ds_config.get("sampling_rate", 16000)
        self.audio_column = ds_config.get("audio_column_name", "audio")
        self.text_column = ds_config.get("text_column_name", "sentence")

        # DataLoader config
        self.num_workers = config.get("num_workers", 4)
        self.batch_size = config.get("batch_size", 16)
        self.prefetch_factor = config.get("prefetch_factor", 2)

        # Tokenizer (will be set from model)
        self.tokenizer = None
        self.vocab_size = 0

        # Datasets
        self.train_dataset: Optional[HFDataset] = None
        self.val_dataset: Optional[HFDataset] = None
        self.test_dataset: Optional[HFDataset] = None

    def prepare_data(self):
        """Download dataset (only called on main process)."""
        # Dataset will be loaded in setup()
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets.

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package is required")

        # Load training dataset
        if stage in (None, "fit"):
            print(f"Loading train dataset: {self.dataset_name}/{self.train_split}")
            self.train_dataset = load_dataset(
                self.dataset_name,
                split=self.train_split,
            )
            if self.max_train_samples:
                self.train_dataset = self.train_dataset.select(range(self.max_train_samples))

            # Apply preprocessing
            self.train_dataset = self.train_dataset.map(
                self._preprocess_audio,
                num_proc=self.num_workers,
                desc="Preprocessing train audio"
            )

        # Load validation dataset
        if stage in (None, "fit", "validate"):
            print(f"Loading eval dataset: {self.dataset_name}/{self.eval_split}")
            self.val_dataset = load_dataset(
                self.dataset_name,
                split=self.eval_split,
            )
            if self.max_eval_samples:
                self.val_dataset = self.val_dataset.select(range(self.max_eval_samples))

            self.val_dataset = self.val_dataset.map(
                self._preprocess_audio,
                num_proc=self.num_workers,
                desc="Preprocessing eval audio"
            )

        # Load test dataset
        if stage in (None, "test"):
            print(f"Loading test dataset: {self.dataset_name}/test")
            self.test_dataset = load_dataset(
                self.dataset_name,
                split="test",
            )
            self.test_dataset = self.test_dataset.map(
                self._preprocess_audio,
                num_proc=self.num_workers,
                desc="Preprocessing test audio"
            )

    def _preprocess_audio(self, batch: dict) -> dict:
        """Preprocess audio: resample and convert to tensor."""
        try:
            import librosa
            import numpy as np
        except ImportError:
            raise ImportError("librosa and numpy are required")

        audio_data = batch[self.audio_column]

        # Handle different audio formats
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio_array = audio_data["array"]
            sr = audio_data.get("sampling_rate", self.sampling_rate)
        else:
            audio_array = audio_data
            sr = self.sampling_rate

        # Resample if needed
        if sr != self.sampling_rate:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sr,
                target_sr=self.sampling_rate
            )

        # Convert to numpy float32
        audio_array = np.array(audio_array, dtype=np.float32)

        # Normalize audio to [-1, 1]
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val

        batch["audio_array"] = audio_array
        batch["audio_length"] = len(audio_array)

        return batch

    def _tokenize_text(self, batch: dict) -> dict:
        """Tokenize text column to token IDs."""
        text = batch.get(self.text_column, "")

        if self.tokenizer is not None:
            # Use model tokenizer
            tokens = self.tokenizer.encode(text)
        else:
            # Fallback: simple character-level tokenization
            # This should be replaced with proper tokenizer
            tokens = [ord(c) % self.vocab_size for c in text]

        batch["token_ids"] = tokens
        batch["token_length"] = len(tokens)

        return batch

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate function to batch samples into tensors.

        Returns:
            Dict with:
                - audio: [batch, max_audio_len]
                - audio_lengths: [batch]
                - tokens: [batch, max_token_len]
                - token_lengths: [batch]
        """
        # Pad audio
        audio_lengths = torch.tensor([item["audio_length"] for item in batch])
        max_audio_len = audio_lengths.max().item()

        audio_batch = torch.zeros(len(batch), max_audio_len)
        for i, item in enumerate(batch):
            audio_array = torch.tensor(item["audio_array"], dtype=torch.float32)
            audio_batch[i, :item["audio_length"]] = audio_array

        # Pad tokens
        token_ids_list = [torch.tensor(item["token_ids"], dtype=torch.long) for item in batch]
        token_lengths = torch.tensor([len(ids) for ids in token_ids_list])
        max_token_len = token_lengths.max().item()

        tokens_batch = torch.zeros(len(batch), max_token_len, dtype=torch.long)
        for i, token_ids in enumerate(token_ids_list):
            tokens_batch[i, :token_lengths[i]] = token_ids

        return {
            "audio": audio_batch,
            "audio_lengths": audio_lengths,
            "tokens": tokens_batch,
            "token_lengths": token_lengths,
        }
