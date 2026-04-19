"""
src/models/anomaly_detector.py
-------------------------------
Bi-GRU Anomaly Detector with MIL ranking head and multi-class classification head.

From: AnomalyDetector_helal_Feb_23.ipynb
"""

import torch
import torch.nn as nn


class AnomalyDetector(nn.Module):
    def __init__(self, input_size: int = 2131, hidden_size: int = 256, num_classes: int = 14):
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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor [Batch, Segments, feature_dim]

        Returns
        -------
        anomaly_scores : Tensor [Batch, Segments, 1]   values in (0, 1)
        class_probs    : Tensor [Batch, Segments, num_classes]  sums to 1
        """
        gru_out, _ = self.bigru(x)
        anomaly_scores = self.anomaly_head(gru_out)
        class_probs = self.class_head(gru_out)
        return anomaly_scores, class_probs
