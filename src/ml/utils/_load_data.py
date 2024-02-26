import torch
from torchvision import transforms
from PIL import Image
import os


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


def load_folder(path: str) -> list[torch.Tensor]:
    images = []
    for filename in os.listdir(path):
        tmp_path = os.path.join(path, filename)
        image = load_file(tmp_path)
        images.append(image)
    return images
