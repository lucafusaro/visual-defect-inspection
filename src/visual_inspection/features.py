from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def extract_image_features(
    image_path: Path,
    model: torch.nn.Module,
    preprocess,
) -> torch.Tensor:
    with Image.open(image_path) as image:
        return extract_pil_image_features(image, model, preprocess)


def extract_pil_image_features(
    image: Image.Image,
    model: torch.nn.Module,
    preprocess,
) -> torch.Tensor:
    image = image.convert("RGB")
    image_tensor = preprocess(image)
    batch = image_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(batch)
    return features


def extract_dataset_features(
    df: pd.DataFrame,
    model: torch.nn.Module,
    preprocess,
) -> torch.Tensor:
    all_features = []

    for image_path in df["path"]:
        features = extract_image_features(image_path, model, preprocess)
        all_features.append(features)

    return torch.cat(all_features, dim=0)


def create_feature_extractor():
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model, preprocess
