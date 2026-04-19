"""
src/models/losses.py
--------------------
MIL (Multiple Instance Learning) Ranking Loss.

From: AnomalyDetector_helal_Feb_23.ipynb
"""

import torch
import torch.nn as nn


class MILRankingLoss(nn.Module):
    """
    MIL Ranking Loss with temporal smoothness and sparsity regularisation.

    Parameters
    ----------
    lambda1 : float
        Weight for temporal smoothness penalty.
    lambda2 : float
        Weight for sparsity penalty (anomaly segments should be rare).
    """

    def __init__(self, lambda1: float = 8e-5, lambda2: float = 8e-5) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, anomaly_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        anomaly_scores : Tensor [Batch, Segments, 1]
        labels         : LongTensor [Batch]   0 = normal, >0 = anomalous

        Returns
        -------
        Tensor scalar loss.
        """
        anomaly_scores = anomaly_scores.squeeze(-1)  # [B, S]

        pos_mask = labels > 0
        neg_mask = labels == 0

        if not pos_mask.any() or not neg_mask.any():
            # Cannot compute ranking loss without both classes present
            return torch.tensor(0.0, device=anomaly_scores.device, requires_grad=True)

        # MIL: max score of anomalous bag vs max score of normal bag
        max_scores, _ = torch.max(anomaly_scores, dim=1)
        max_anomaly = max_scores[pos_mask].mean()
        max_normal = max_scores[neg_mask].mean()

        # Hinge ranking loss (margin = 1)
        ranking_loss = torch.relu(1.0 - max_anomaly + max_normal)

        # Temporal smoothness: penalise large jumps between consecutive segments
        diff = anomaly_scores[:, 1:] - anomaly_scores[:, :-1]
        smoothness = torch.sum(diff ** 2)

        # Sparsity: anomaly segments should be rare
        sparsity = torch.sum(anomaly_scores[pos_mask])

        return ranking_loss + self.lambda1 * smoothness + self.lambda2 * sparsity
