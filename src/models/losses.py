"""
Loss functions for anomaly detection training.

Implements MIL (Multiple Instance Learning) ranking loss from the notebook.
"""

import torch
import torch.nn as nn


class MILRankingLoss(nn.Module):
    def __init__(self, lambda1=8e-5, lambda2=8e-5):
        super(MILRankingLoss, self).__init__()
        self.lambda1 = lambda1  # Temporal Smoothness weight
        self.lambda2 = lambda2  # Sparsity weight

    def forward(self, anomaly_scores, labels):
        anomaly_scores = anomaly_scores.squeeze(-1)

        # Identify Anomalous vs Normal videos in the current batch
        pos_mask = (labels > 0)
        neg_mask = (labels == 0)

        # Safety check for MIL ranking requirement
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=anomaly_scores.device, requires_grad=True)

        # MIL Ranking: Compare max score of Anomaly video vs max score of Normal video
        max_scores, _ = torch.max(anomaly_scores, dim=1)
        max_anomaly = max_scores[pos_mask].mean()
        max_normal = max_scores[neg_mask].mean()

        # Hinge loss to enforce a margin of 1.0
        ranking_loss = torch.relu(1.0 - max_anomaly + max_normal)

        # Smoothness: Penalize large score jumps between consecutive segments
        diff = anomaly_scores[:, 1:] - anomaly_scores[:, :-1]
        smoothness = torch.sum(diff**2)

        # Sparsity: Enforce that anomaly segments are rare (short-duration)
        sparsity = torch.sum(anomaly_scores[pos_mask])

        return ranking_loss + (self.lambda1 * smoothness) + (self.lambda2 * sparsity)
