import torch
import torchvision
import clip
import torch.nn as nn

from torchvision import transforms
from torchvision.transforms import InterpolationMode
num_classes = 5

vit_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation= InterpolationMode.BICUBIC),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation = 0.2, hue = 0.05),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.Normalize(
        mean = [0.48145466, 0.4578275, 0.40821073],
        std  = [0.26862954, 0.26130258, 0.27577711])
    ])


vit_val_transform = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224 ,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
backbone, preprocess = clip.load("ViT-B/32", device = device, jit = False)
backbone.eval()

train_transforms = preprocess

for param in backbone.parameters():
    param.requires_grad = False

classification = nn.Linear(512, num_classes).to(device)