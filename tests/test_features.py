import torch
from PIL import Image

from visual_inspection.features import extract_pil_image_features


def test_extract_pil_image_features_converts_image_to_rgb():
    image = Image.new("L", (2, 2), color=128)

    def preprocess(pil_image):
        assert pil_image.mode == "RGB"
        return torch.ones((3, 2, 2))

    features = extract_pil_image_features(image, torch.nn.Identity(), preprocess)

    assert features.shape == (1, 3, 2, 2)
