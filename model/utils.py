import torch
from torchvision import transforms

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_transforms():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
