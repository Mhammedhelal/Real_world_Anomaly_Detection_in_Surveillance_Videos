import torchvision.transforms as transforms
from src.config import Config
from pathlib import Path

# Resolve config path relative to project root
config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'default.yaml'
cfg = Config.from_yaml(config_path)

# Get frame_size and handle both tuple and string formats
frame_size_raw = getattr(cfg.dataset, 'frame_size', (224, 224))
if isinstance(frame_size_raw, str):
    # Parse string like "(224, 224)" to tuple
    import ast
    try:
        frame_size = ast.literal_eval(frame_size_raw)
    except:
        frame_size = (224, 224)
else:
    frame_size = frame_size_raw

# Get normalization parameters with fallbacks
mean = getattr(cfg.dataset, 'mean', [0.485, 0.456, 0.406])
std = getattr(cfg.dataset, 'std', [0.229, 0.224, 0.225])

# For backward compatibility
IMAGE_SIZE = frame_size

transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(frame_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,
                                    std=std)
                ])