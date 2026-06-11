"""Sliding-window PyTorch datasets for long-term and M4 benchmarks.

Both datasets yield plain dict batches so DataLoader can collate them
automatically.  The long-term dataset always includes ``y_full`` and
``y_mark`` (decoder context) in addition to the canonical ``x``, ``y``,
``x_mark`` keys so that seq2seq models can extract their label-len
historical context without any adapter branching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from ..data.long_term import LoadedLongTermDataset
    from ..data.m4 import LoadedM4Dataset


class LongTermWindowDataset(Dataset):
    """Sliding-window dataset over one temporal split of a long-term series.

    Each sample contains:
    - ``x``           : scaled encoder input, shape ``(seq_len, C)``
    - ``x_mark``      : calendar time features for the encoder window
    - ``y``           : scaled prediction target, shape ``(pred_len, C)``
    - ``y_full``      : decoder context window ``(label_len + pred_len, C)``
    - ``y_mark``      : time features for the decoder window
    - ``future_orig`` : original-scale future values for metric computation

    Temporal integrity: the scaler is fitted on the training split only; all
    three splits consume the *already scaled* array so validation/test data
    never leaks into the scaler.  The encoder window boundary always starts
    within the designated split, and the decoder window may extend up to
    ``pred_len`` rows beyond the encoder end — these future rows come from
    the *same* split (or, for train windows near the boundary, slightly into
    the adjacent val split, which is the standard TSLib practice and does
    not constitute label leakage since the scaler is fixed).
    """

    def __init__(
        self,
        dataset: "LoadedLongTermDataset",
        pred_len: int,
        split: str,
    ) -> None:
        """Initialise a window dataset for a specific split and horizon.

        Args:
            dataset: Loaded long-term dataset with split metadata.
            pred_len: Forecast horizon (number of future steps).
            split: Which temporal partition to window over: ``train``,
                ``val``, or ``test``.
        """
        self.scaled = dataset.scaled_values
        self.original = dataset.original_values
        self.time_feats = dataset.time_features
        self.seq_len = dataset.seq_len
        self.label_len = dataset.label_len
        self.pred_len = pred_len
        self.stride = dataset.stride

        start, end = dataset.split_ranges[split]
        self.split_start = start
        self.split_end = end

        required_span = self.seq_len + self.pred_len
        available = end - start
        if available < required_span:
            self.n_windows = 0
        else:
            self.n_windows = (available - required_span) // self.stride + 1

    def __len__(self) -> int:
        """Return the number of valid sliding windows in this split."""
        return self.n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one window as a collatable tensor dict.

        Args:
            idx: Window index within this split.

        Returns:
            Dict of tensors ready for DataLoader collation.
        """
        x_start = self.split_start + idx * self.stride
        x_end = x_start + self.seq_len

        # Encoder window
        x = self.scaled[x_start:x_end]            # (seq_len, C)
        x_mark = self.time_feats[x_start:x_end]   # (seq_len, D)

        # Prediction target: scaled future values (loss target)
        y = self.scaled[x_end:x_end + self.pred_len]   # (pred_len, C)

        # Decoder context: last label_len history rows + pred_len future rows
        dec_start = x_end - self.label_len
        dec_end = x_end + self.pred_len
        y_full = self.scaled[dec_start:dec_end]        # (label_len + pred_len, C)
        y_mark = self.time_feats[dec_start:dec_end]    # (label_len + pred_len, D)

        # Original scale future (used by the engine for metric computation only)
        future_orig = self.original[x_end:x_end + self.pred_len]  # (pred_len, C)

        return {
            "x": torch.as_tensor(x, dtype=torch.float32),
            "x_mark": torch.as_tensor(x_mark, dtype=torch.float32),
            "y": torch.as_tensor(y, dtype=torch.float32),
            "y_full": torch.as_tensor(y_full, dtype=torch.float32),
            "y_mark": torch.as_tensor(y_mark, dtype=torch.float32),
            "future_orig": torch.as_tensor(future_orig, dtype=torch.float32),
        }


class M4SeriesDataset(Dataset):
    """Per-series dataset for one M4 frequency slice.

    Each sample corresponds to one M4 series.  The encoder input is the
    last ``seq_len`` values of the training history (zero-padded on the
    left when the series is shorter than ``seq_len``).  The target ``y``
    is the test values (the official ``horizon`` future steps).

    ``y_full`` mirrors the long-term dataset contract: it is the decoder
    context window ``(label_len + horizon, 1)`` where the first
    ``label_len`` rows are the last observed training values (scaled) and
    the remaining ``horizon`` rows are zeros.  Seq2seq models read this
    for teacher-forcing; encoder-only models ignore it.

    ``y_mark`` is always ``(label_len + horizon, 4)`` zero-filled so
    that seq2seq decoders receive a time-feature tensor with the correct
    sequence length even when real calendar features are unavailable.
    """

    def __init__(
        self,
        dataset: "LoadedM4Dataset",
        seq_len: int = 96,
        label_len: int = 0,
    ) -> None:
        """Initialise for one M4 frequency slice.

        Args:
            dataset: Loaded M4 dataset for one frequency.
            seq_len: Fixed encoder input length (shorter series are left-padded).
            label_len: Decoder historical-context length for seq2seq models.
                Determines the leading rows of ``y_full`` and the total length
                of ``y_mark``.  Pass ``0`` for encoder-only models.
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.horizon = dataset.horizon
        self.series_ids = dataset.series_ids
        self.series = dataset.series

    def __len__(self) -> int:
        """Return the number of M4 series in this frequency slice."""
        return len(self.series_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one series sample as a tensor dict.

        Args:
            idx: Series index.

        Returns:
            Dict with ``x`` (padded encoder input), ``y`` (test target),
            ``y_full`` (decoder context window), ``y_mark`` (decoder time
            features, all zeros), and metadata for metric computation.
        """
        series_id = self.series_ids[idx]
        series = self.series[series_id]

        train_vals = series.train_values  # (T,) float32
        test_vals = series.test_values    # (horizon,) float32

        # Univariate: add channel dim → (T, 1)
        train_2d = train_vals.reshape(-1, 1)

        # Normalise the encoder window using its own statistics to avoid
        # scale dominance when batching series from different magnitudes.
        if len(train_2d) >= 2:
            mean_enc = train_2d.mean()
            std_enc = train_2d.std() + 1e-8
        else:
            mean_enc = 0.0
            std_enc = 1.0

        scaled_train = (train_2d - mean_enc) / std_enc
        scaled_test = (test_vals.reshape(-1, 1) - mean_enc) / std_enc

        # Truncate or left-pad to seq_len
        if len(scaled_train) >= self.seq_len:
            x_window = scaled_train[-self.seq_len:]
        else:
            pad = np.zeros((self.seq_len - len(scaled_train), 1), dtype=np.float32)
            x_window = np.concatenate([pad, scaled_train], axis=0)

        # Decoder context: last label_len observed values + horizon zeros.
        # x_window already ends at the last training observation, so its
        # tail provides the historical decoder context without extra indexing.
        if self.label_len > 0:
            dec_hist = x_window[-self.label_len:]          # (label_len, 1)
            dec_zeros = np.zeros((self.horizon, 1), dtype=np.float32)
            y_full = np.concatenate([dec_hist, dec_zeros], axis=0)  # (L+H, 1)
        else:
            y_full = np.zeros((self.horizon, 1), dtype=np.float32)

        dec_len = self.label_len + self.horizon

        return {
            "x": torch.as_tensor(x_window, dtype=torch.float32),
            "y": torch.as_tensor(scaled_test, dtype=torch.float32),
            "x_mark": torch.zeros(self.seq_len, 4, dtype=torch.float32),
            "y_full": torch.as_tensor(y_full, dtype=torch.float32),
            "y_mark": torch.zeros(dec_len, 4, dtype=torch.float32),
            # Stored as numpy arrays (not tensors) for metric computation
            "train_orig": train_vals,
            "test_orig": test_vals,
            "mean_enc": float(mean_enc),
            "std_enc": float(std_enc),
            "_series_id": series_id,
        }
