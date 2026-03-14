import torchvision.transforms as transforms
from src.config import Config
from pathlib import Path
# Resolve config path relative to project root
config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'default.yaml'
cfg = Config.from_yaml(config_path)
IMAGE_SIZE =  cfg.dataset.image_size

mean = cfg.dataset.mean
std  = cfg.dataset.std
frame_size = cfg.dataset.frame_size

transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(frame_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,
                                    std=std)
                ])