import os
import cv2
import torch
import timm
import numpy as np
from glob import glob
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import monai.transforms as M
import torch.serialization

pixels = 320
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

resize_val = M.Compose([
    M.EnsureChannelFirst(channel_dim=2),
    M.CropForeground(),
    M.Resize(pixels, size_mode='longest'),
    M.SpatialPad((pixels, pixels)),
    M.AsChannelLast()
])

def get_transforms():
    return A.Compose([ToTensorV2()])

class CustomResNet(nn.Module):
    def __init__(self, name="resnet50", pretrained=False):
        super().__init__()
        assert name in ['resnet18', 'resnet34', 'resnet50'], "Only ResNet backbones supported"
        self.model = timm.create_model(name, pretrained=pretrained, in_chans=3)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.model(x)

def preprocess_image(img_np):
    img = torch.from_numpy(img_np[..., np.newaxis].astype(np.float32)) / 255.

    image3ch = np.zeros((pixels, pixels, 3), dtype=np.float32)
    resized = resize_val(img)[..., 0]
    image3ch[..., 0] = image3ch[..., 1] = image3ch[..., 2] = resized

    transformed = get_transforms()(image=image3ch)["image"]
    return transformed.to(device)



def load_model_ensemble(gender, ensemble_dir):
    pattern = os.path.join(ensemble_dir, f"{'male' if gender == 0 else 'female'}_resnet50_fold*_best_score.pth")
    state_paths = sorted(glob(pattern))

    models = []
    for path in state_paths:
        model = CustomResNet("resnet50", pretrained=False).to(device)

        with torch.serialization.safe_globals(['numpy.core.multiarray._reconstruct']):
            state = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
        model.eval()
        models.append(model)

    return models



def predict(img_np, gender_str, ensemble_dir):
    gender_str = gender_str.strip().upper()
    if gender_str not in ["M", "F"]:
        raise ValueError(f"Unexpected gender string: {gender_str}")

    gender = 0 if gender_str == "M" else 1
    image = preprocess_image(img_np)
    models = load_model_ensemble(gender, ensemble_dir)

    with torch.no_grad():
        preds = [model(image.unsqueeze(0)).cpu().numpy() for model in models]

    avg_pred = np.mean(preds) * 200
    return round(float(avg_pred), 2)


