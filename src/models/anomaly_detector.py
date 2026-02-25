"""
Anomaly Detection Models

This module contains the AnomalyDetector model from the notebook.
"""

import torch
import torch.nn as nn
from typing import Tuple


class AnomalyDetector(nn.Module):
    def __init__(self, input_size=2131, hidden_size=256, num_classes=14):
        super(AnomalyDetector, self).__init__()

        # 1. Temporal Encoder: Processes fused I3D (2048) + YOLO (83) features
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Bi-directional output is concatenated (hidden_size * 2)
        combined_dim = hidden_size * 2

        # 2. MIL Ranking Head: Produces a [0, 1] anomaly score per segment
        self.anomaly_head = nn.Sequential(
            nn.Linear(combined_dim, 1),
            nn.Sigmoid()
        )

        # 3. Multi-Class Classification Head: Identifies specific crime category
        self.class_head = nn.Sequential(
            nn.Linear(combined_dim, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x shape: [Batch, Segments, 2131]
        gru_out, _ = self.bigru(x)

        # Branch A: Regression-like score for temporal localization
        anomaly_scores = self.anomaly_head(gru_out)

        # Branch B: Fine-grained classification probabilities
        class_probs = self.class_head(gru_out)

        return anomaly_scores, class_probs
