# Copilot Instructions for Real-world Anomaly Detection in Surveillance Videos

## Project Overview

This is a machine learning project for detecting anomalies in surveillance videos using the UCF-Crime dataset. The system implements a Multiple Instance Learning (MIL) approach with feature extraction using I3D (Inflated 3D ConvNet) and YOLO (You Only Look Once) object detection models.

**Key Goals:**
- Extract spatio-temporal features from video frames
- Train an anomaly detection model using MIL ranking loss
- Evaluate performance on anomalous vs normal video segments
- Provide real-time anomaly detection capabilities

## Technology Stack

- **Core Framework:** PyTorch with torchvision/torchaudio
- **Computer Vision:** OpenCV for video processing
- **Object Detection:** Ultralytics YOLOv8
- **Configuration:** YAML-based configuration system
- **Data Processing:** NumPy, custom dataset classes
- **Visualization:** Matplotlib, Seaborn
- **Video I/O:** MoviePy or FFmpeg

## Project Structure & Key Files

```
├── configs/default.yaml          # Main configuration file
├── src/
│   ├── config.py                 # Configuration management class
│   ├── data/
│   │   ├── dataset.py            # VideoFeatureDataset class
│   │   ├── transforms.py         # Data transformations
│   │   └── metadata.py           # Dataset metadata handling
│   ├── models/
│   │   ├── anomaly_detector.py   # Main MIL model architecture
│   │   ├── feature_extractors.py # I3D and YOLO feature extractors
│   │   ├── losses.py             # MILRankingLoss implementation
│   │   └── video_preprocessor.py # Video preprocessing utilities
│   ├── engine/
│   │   ├── trainer.py            # Training loop and logic
│   │   └── FeatureExtractionPipeline.py # Feature extraction pipeline
│   └── utils/
│       ├── metrics.py            # Evaluation metrics
│       ├── checkpointing.py      # Model saving/loading
│       ├── video.py              # Video processing utilities
│       └── visualization.py      # Plotting and visualization
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation script
│   ├── predict.py                # Inference script
│   └── extract_features.py       # Feature extraction script
└── notebooks/                    # Jupyter notebooks for experimentation
```

## Code Style & Conventions

### Python Style
- Follow PEP 8 conventions
- Use type hints for function parameters and return values
- Add docstrings to all classes and functions using Google/NumPy style
- Use descriptive variable names (avoid single letters except for loops)

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import torch
import numpy as np
import yaml

# Local imports (grouped by module)
from src.config import Config
from src.data.dataset import VideoFeatureDataset
from src.models.anomaly_detector import AnomalyDetector
```

### Configuration Management
- Use the `Config` class from `src/config.py` for all configuration
- Access config values using dot notation: `cfg.training.batch_size`
- Override config values programmatically when needed
- Save modified configs to output directories

## Common Patterns & Practices

### 1. Model Architecture
```python
class AnomalyDetector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)
```

### 2. Data Loading
```python
from torch.utils.data import DataLoader

dataset = VideoFeatureDataset(
    features_dir="data/features",
    split_file="data/splits/train.txt",
    config=cfg
)

dataloader = DataLoader(
    dataset,
    batch_size=cfg.training.batch_size,
    shuffle=True,
    num_workers=cfg.training.num_workers,
    collate_fn=collate_fn
)
```

### 3. Training Loop
```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (features, labels, metadata) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### 4. Feature Extraction
```python
# I3D feature extraction
i3d_model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
i3d_model.eval()

# YOLO object detection
from ultralytics import YOLO
yolo_model = YOLO('yolov8n.pt')
```

## Configuration Schema

Key configuration sections in `configs/default.yaml`:

- **dataset**: Video directories, frame processing, extensions
- **feature_extraction**: I3D/YOLO parameters, normalization
- **model**: Network architecture (input_size, hidden_size, num_classes)
- **training**: Batch size, epochs, learning rate, optimizer settings
- **loss**: MIL loss parameters (lambda_smoothness, lambda_sparsity)
- **hardware**: Device settings, mixed precision
- **logging**: Intervals, checkpoint directories

## Testing Approach

- Unit tests in `tests/` directory
- Test model components, data loading, metrics
- Use pytest framework
- Mock external dependencies when possible

## Performance Considerations

- Use GPU acceleration for training and feature extraction
- Implement mixed precision training when possible
- Batch processing for video feature extraction
- Memory-efficient data loading with `pin_memory=True`

## Error Handling

- Check for CUDA availability before GPU operations
- Handle missing video files gracefully
- Validate configuration parameters
- Log errors with context information

## File I/O Patterns

- Use `pathlib.Path` for path operations
- Create output directories automatically
- Save checkpoints with timestamps
- Use JSON/YAML for metadata storage

## Debugging Tips

- Use tensorboard or wandb for training visualization
- Print model summaries with `torchsummary`
- Check tensor shapes and dtypes during development
- Use `torch.cuda.empty_cache()` for memory issues

## Deployment Considerations

- Export models to ONNX for inference optimization
- Containerize with Docker for reproducibility
- Implement model versioning
- Consider edge deployment for real-time video processing

## AI Assistant Guidelines

When working on this project:

1. **Understand the MIL context**: This is Multiple Instance Learning where bags contain instances, and we predict bag-level labels
2. **Respect the config system**: Always use Config class for parameters
3. **Follow PyTorch best practices**: Proper device handling, gradient management
4. **Maintain video processing pipeline**: Features → Model → Predictions
5. **Consider computational constraints**: Videos are memory-intensive
6. **Test thoroughly**: ML bugs can be subtle and hard to detect
7. **Document assumptions**: Video formats, frame rates, resolutions

## Common Tasks

- **Adding new features**: Update config, implement in appropriate module, add tests
- **Model modifications**: Change architecture in `models/`, update config defaults
- **Data processing**: Modify `data/` modules, ensure compatibility with existing pipeline
- **Training improvements**: Update `engine/trainer.py`, experiment in notebooks first
- **Evaluation metrics**: Add to `utils/metrics.py`, update evaluation scripts