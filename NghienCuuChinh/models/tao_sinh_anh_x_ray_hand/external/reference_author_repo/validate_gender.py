import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import monai.transforms as M
import timm
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

class CFG:
    img_size = 320
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble_size = 5
    model_dir = r"C:\Projetos\hand-bone\gender"
    test_img_dir = r"C:\Projetos\hand-bone\output_cleaned_mult"
    test_csv = r"C:\Projetos\hand-bone\boneage-test-dataset.csv"
    out_csv = r"C:\Projetos\hand-bone\ensemble_gender_predictions_mult.csv"

resize_val = M.Compose([
    M.EnsureChannelFirst(channel_dim=-1),
    M.CropForeground(),
    M.Resize(CFG.img_size, size_mode='longest'),
    M.SpatialPad((CFG.img_size, CFG.img_size)),
    M.AsChannelLast(),
])

def get_transforms():
    return A.Compose([
        ToTensorV2(),
    ])

class CustomModel(nn.Module):
    def __init__(self, name='resnet50'):
        super().__init__()
        self.model = timm.create_model(name, pretrained=False, in_chans=3)
        if name in ['resnet50','resnet34','resnet18']:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 1)
    def forward(self, x, gender=None):
        x = self.model(x)
        return x

def preprocess_img(img_path):
    img = np.array(Image.open(img_path).convert('L'))
    img = np.stack([img, img, img], axis=-1)
    img = resize_val(img)
    img = img.astype(np.float32) / 255.
    transforms = get_transforms()
    img = transforms(image=img)['image']
    return img

def load_ensemble(model_dir, device):
    models = []
    for k in range(CFG.ensemble_size):
        model = CustomModel('resnet50')
        weight_path = os.path.join(model_dir, f"resnet50_gender_{k}.pth")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models

def infer_ensemble(models, img_tensor):
    preds = []
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(CFG.device)
    with torch.no_grad():
        for model in models:
            logit = model(img_tensor)
            prob = torch.sigmoid(logit).cpu().numpy().flatten()[0]
            preds.append(prob)
    mean_prob = np.mean(preds)
    return mean_prob

if __name__ == "__main__":
    models = load_ensemble(CFG.model_dir, CFG.device)
    df = pd.read_csv(CFG.test_csv)
    df.columns = [c.strip() for c in df.columns]
    if "id" not in df.columns:
        for candidate in ["Case ID", "case_id"]:
            if candidate in df.columns:
                df["id"] = df[candidate]
                break

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Ensemble Inference"):
        img_id = str(row["id"])
        patient_dir = os.path.join(CFG.test_img_dir, img_id)
        if not os.path.exists(patient_dir):
            print(f"Patient folder not found: {patient_dir}")
            continue
        # Find all image files for this patient
        img_files = [f for f in os.listdir(patient_dir) if f.startswith(f"{img_id}_cleaned_") and f.endswith(".png")]
        if not img_files:
            print(f"No images found for patient: {img_id}")
            continue
        for img_file in img_files:
            img_path = os.path.join(patient_dir, img_file)
            try:
                img_tensor = preprocess_img(img_path)
                prob = infer_ensemble(models, img_tensor)
                pred_label = int(prob > 0.5)
                results.append({
                    "id": img_id,
                    "image_name": img_file,
                    "prob_male": prob,
                    "predicted_label": pred_label,
                })
            except Exception as e:
                print(f"[ERROR] on {img_path}: {e}")

    pd.DataFrame(results).to_csv(CFG.out_csv, index=False)
    print(f"Predictions saved to: {CFG.out_csv}")
