import torch
from torchvision import transforms
from PIL import Image


val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_file(path: str) -> torch.Tensor:
    image = Image.open(path)
    image = val_transforms(image)
    return image
