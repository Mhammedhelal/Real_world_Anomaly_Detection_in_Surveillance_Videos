"""
Training entrypoint for the UCF-Crime Anomaly Detector.

Usage (Colab):
    %run train.py --features-dir /content/drive/MyDrive/UCF_Crime/features --epochs 100

Or import and call directly:
    from train import train
    train(features_dir=..., save_dir=..., epochs=100)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Model imports (adjust path if running from a subdirectory) ──────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.anomaly_detector import AnomalyDetector
from src.models.losses import MILRankingLoss


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class VideoFeatureDataset(Dataset):
    """
    Loads pre-extracted .npz feature files produced by extract_features.py.

    Each .npz file must contain:
        - 'features'  : np.ndarray  [num_segments, feature_dim]
        - 'metadata'  : dict-like   with key 'label' (int, 0 = normal)
    """

    def __init__(self, features_dir: str, split: str = "train"):
        """
        Args:
            features_dir: Root features directory (contains subfolders).
            split: 'train' or 'test' — files whose names start with this prefix.
        """
        self.samples = []   # list of (features_tensor, label)
        self.filenames = []

        # Walk all subfolders looking for .npz files matching the split
        for root, _, files in os.walk(features_dir):
            for fname in sorted(files):
                if not fname.endswith(".npz"):
                    continue
                if not fname.startswith(split + "_"):
                    continue

                path = os.path.join(root, fname)
                try:
                    data = np.load(path, allow_pickle=True)
                    features = data["features"].astype(np.float32)    # [S, D]
                    metadata = data["metadata"].item()
                    label = int(metadata["label"])

                    self.samples.append((torch.from_numpy(features), label))
                    self.filenames.append(fname)
                except Exception as e:
                    print(f"⚠️  Skipping {fname}: {e}")

        print(f"📦 Loaded {len(self.samples)} '{split}' feature files from {features_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return features, label


def collate_fn(batch):
    """Pad variable-length segment sequences to the longest in the batch."""
    features, labels = zip(*batch)
    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = torch.LongTensor(labels)
    return features_padded, labels


# ════════════════════════════════════════════════════════════════════════════
# Training function
# ════════════════════════════════════════════════════════════════════════════

def train(
    features_dir: str,
    save_dir: str = "./checkpoints",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1.0,
    input_size: int = 2131,
    hidden_size: int = 256,
    num_classes: int = 14,
    lambda1: float = 8e-5,
    lambda2: float = 8e-5,
    save_every: int = 10,
    device: str = None,
):
    """
    Full training loop.

    Args:
        features_dir : Directory that contains .npz feature files.
        save_dir     : Where to save model checkpoints.
        epochs       : Number of training epochs.
        batch_size   : Batch size for DataLoader.
        lr           : Learning rate for Adadelta.
        input_size   : Feature vector dimension (I3D 2048 + YOLO 83 = 2131).
        hidden_size  : Bi-GRU hidden size.
        num_classes  : Number of crime categories (14 for UCF-Crime).
        lambda1      : Smoothness regularisation weight.
        lambda2      : Sparsity regularisation weight.
        save_every   : Save a checkpoint every N epochs.
        device       : 'cuda' | 'cpu' | None (auto-detect).

    Returns:
        model        : Trained AnomalyDetector.
        loss_history : List of per-epoch average losses.
    """

    # ── Device ──────────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"🖥️  Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    dataset = VideoFeatureDataset(features_dir, split="train")
    if len(dataset) == 0:
        raise RuntimeError(
            f"No 'train_*.npz' files found in {features_dir}. "
            "Run extract_features.py first."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,   # 0 is safest for Colab
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = AnomalyDetector(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
    ).to(device)

    print(f"🧠 AnomalyDetector | input={input_size} | hidden={hidden_size} | classes={num_classes}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {total_params:,}")

    # ── Optimiser & losses ──────────────────────────────────────────────────
    # Adadelta is the original choice from the notebook; robust to sparse gradients.
    optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-6)

    criterion_mil   = MILRankingLoss(lambda1=lambda1, lambda2=lambda2)
    criterion_class = nn.CrossEntropyLoss()

    # ── Checkpoint directory ─────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for features, labels in loader:
            features = features.to(device)
            labels   = labels.to(device)

            # Forward
            anomaly_scores, class_logits = model(features)

            # MIL ranking loss
            loss_mil = criterion_mil(anomaly_scores, labels)

            # Classification loss: flatten time axis → [B*S, C]
            num_segments = features.size(1)
            # Repeat each video label for every segment
            labels_expanded = labels.repeat_interleave(num_segments)
            loss_cls = criterion_class(
                class_logits.view(-1, num_classes),
                labels_expanded,
            )

            total_loss = loss_mil + loss_cls

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss  += total_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:>4}/{epochs}]  Loss: {avg_loss:.4f}")

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(save_dir, f"anomaly_detector_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": {
                        "input_size": input_size,
                        "hidden_size": hidden_size,
                        "num_classes": num_classes,
                    },
                },
                ckpt_path,
            )
            print(f"💾 Checkpoint saved → {ckpt_path}")

    print(f"\n✅ Training complete. Best loss: {min(loss_history):.4f}")
    return model, loss_history


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train the UCF-Crime anomaly detector.")
    p.add_argument("--features-dir", type=str, required=True,
                   help="Root directory containing .npz feature files.")
    p.add_argument("--save-dir", type=str, default="./checkpoints",
                   help="Directory to save model checkpoints.")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1.0)
    p.add_argument("--input-size",  type=int,   default=2131,
                   help="Feature vector dimension (default: 2048 I3D + 83 YOLO = 2131).")
    p.add_argument("--hidden-size", type=int,   default=256)
    p.add_argument("--num-classes", type=int,   default=14)
    p.add_argument("--save-every",  type=int,   default=10,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--device", type=str, default=None,
                   help="'cuda' or 'cpu'. Auto-detected if not set.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, history = train(
        features_dir=args.features_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
        save_every=args.save_every,
        device=args.device,
    )

    # Quick loss plot (no dependency on src.utils)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(history, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "training_loss.png"), dpi=150)
        print("📈 Loss plot saved.")
        plt.show()
    except Exception:
        pass