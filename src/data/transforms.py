import torchvision.transforms as transforms


class Transform:
    def build_transform(frame_size, mean, std):
        return transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(frame_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,
                                    std=std)
                ])
    

mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]