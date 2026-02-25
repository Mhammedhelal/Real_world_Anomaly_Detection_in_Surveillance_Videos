"""
Training loop and trainer utilities for anomaly detection model.

From: AnomalyDetector_helal_Feb_23.ipynb
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Trainer class for anomaly detection model"""

    def __init__(self, model, train_loader, device='cuda', learning_rate=1.0, 
                 num_epochs=100, rho=0.9, eps=1e-6):
        """
        Initialize trainer
        
        Args:
            model: AnomalyDetector model
            train_loader: DataLoader for training data
            device: Device to train on
            learning_rate: Learning rate for Adadelta
            num_epochs: Number of training epochs
            rho: Rho parameter for Adadelta
            eps: Epsilon parameter for Adadelta
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Optimizer: Adadelta is robust for MIL tasks involving sparse gradients
        self.optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=learning_rate,
            rho=rho,
            eps=eps
        )
        
        # Loss functions
        from ..models import MILRankingLoss
        self.criterion_mil = MILRankingLoss(lambda1=8e-5, lambda2=8e-5)
        self.criterion_class = nn.CrossEntropyLoss()

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0

        for features, labels in self.train_loader:
            features, labels = features.to(self.device), labels.to(self.device)

            # Forward pass: Generate detection scores and classification logits
            anomaly_scores, class_logits = self.model(features)

            # Multi-task loss calculation
            loss_ranking = self.criterion_mil(anomaly_scores, labels)

            # Flatten time dimension for cross-entropy: [Batch*Segments, Classes]
            loss_classification = self.criterion_class(
                class_logits.view(-1, 14),
                labels.repeat_interleave(features.size(1))
            )

            total_loss = loss_ranking + loss_classification

            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            epoch_loss += total_loss.item()

        return epoch_loss / len(self.train_loader)

    def train(self):
        """Train the model"""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        for epoch in range(self.num_epochs):
            epoch_loss = self.train_epoch()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")

        print(f"\n✅ Training completed!")
        return self.model


def train_model(model, all_features, all_labels, batch_size=32, num_epochs=100, 
                device='cuda', learning_rate=1.0, rho=0.9, eps=1e-6):
    """
    Train anomaly detection model
    
    Args:
        model: AnomalyDetector model
        all_features: List of feature tensors
        all_labels: List of labels
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        device: Device to train on
        learning_rate: Learning rate for optimizer
        rho: Rho parameter for Adadelta
        eps: Epsilon parameter for Adadelta
        
    Returns:
        Trained model
    """
    from ..data import VideoDataset, collate_fn_variable_length
    
    # Create dataset and dataloader
    dataset = VideoDataset(all_features, all_labels)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_variable_length
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        rho=rho,
        eps=eps
    )

    # Train model
    return trainer.train()
