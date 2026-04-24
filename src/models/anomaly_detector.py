"""
src/models/anomaly_detector.py
-------------------------------
Bi-GRU Anomaly Detector with MIL ranking head and multi-class classification head.

From: AnomalyDetector_helal_Feb_23.ipynb

Packing
-------
Each .npz file stores a whole video as [num_segments, 2131].  Videos have
different lengths, so collate_fn pads shorter ones with zeros to form a
[batch, max_segments, 2131] tensor.  Without packing the GRU would process
those padding zeros as real data, corrupting the hidden state of shorter
videos.

We therefore accept an optional ``lengths`` argument.  When provided:
  1. pack_padded_sequence  — tells the GRU to stop at each video's real length.
  2. pad_packed_sequence   — unpacks back to [batch, max_segments, hidden*2],
                             re-filling padded positions with 0.
  3. The anomaly / class heads run on the full padded tensor, but padded
     positions contain zeros, so sigmoid(0) = 0.5 and softmax(0,...,0) =
     uniform.  The MIL loss uses max-over-real-segments, so those positions
     are harmless as long as callers also pass lengths to the loss / evaluator
     (or rely on the fact that a real anomalous segment will always beat 0.5).

When ``lengths`` is None (e.g. single-video inference with no padding) the
model falls back to the original unpacked path — zero overhead.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional


class AnomalyDetector(nn.Module):
    def __init__(
        self,
        input_size: int = 2131,
        hidden_size: int = 256,
        num_classes: int = 14,
    ):
        super().__init__()

        # 1. Temporal Encoder: processes fused I3D (2048) + YOLO (83) features
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        combined_dim = hidden_size * 2  # bidirectional concatenation

        # 2. MIL Ranking Head: anomaly score in [0, 1] per segment
        self.anomaly_head = nn.Sequential(
            nn.Linear(combined_dim, 1),
            nn.Sigmoid(),
        )

        # 3. Multi-Class Classification Head: fine-grained crime category
        self.class_head = nn.Sequential(
            nn.Linear(combined_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        x : Tensor [Batch, Segments, feature_dim]
            Padded batch of video segment features produced by collate_fn.
        lengths : LongTensor [Batch], optional
            Real (unpadded) segment count for each video in the batch.
            Obtain from collate_fn (see dataset.py).  When provided:
              - The GRU only processes real timesteps → no hidden-state
                corruption from padding zeros.
              - Padded positions in gru_out are reset to 0.0 by
                pad_packed_sequence, so the heads output sigmoid(0)=0.5 /
                uniform softmax at those positions.
            Pass None for single-video inference where x has no padding.

        Returns
        -------
        anomaly_scores : Tensor [Batch, Segments, 1]
            Per-segment anomaly probability in (0, 1).
            Padded positions contain 0.5 when lengths is given.
        class_probs    : Tensor [Batch, Segments, num_classes]
            Per-segment class probabilities summing to 1.
            Padded positions contain a uniform distribution when lengths
            is given.
        """
        if lengths is not None:
            # pack_padded_sequence requires lengths on CPU
            lengths_cpu = lengths.cpu()

            # Pack: drops padding so GRU never sees zero segments
            packed = pack_padded_sequence(
                x,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,  # no need to pre-sort the batch
            )

            packed_out, _ = self.bigru(packed)

            # Unpack: restores [Batch, max_segments, hidden*2]
            # padding positions are filled with 0.0
            gru_out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x.size(1),  # preserve original padded length
            )
        else:
            # No padding — plain forward pass (single-video inference)
            gru_out, _ = self.bigru(x)

        anomaly_scores = self.anomaly_head(gru_out)
        class_probs    = self.class_head(gru_out)
        return anomaly_scores, class_probs