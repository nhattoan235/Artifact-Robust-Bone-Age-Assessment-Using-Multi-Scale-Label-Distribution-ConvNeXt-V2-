import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
import monai.transforms as M
import warnings
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

class CFG:
    img_size = 320
    batch_size = 32
    epochs = 5
    lr = 1e-4
    n_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = r'C:\Projetos\hand-bone\boneage-training-dataset\boneage-training-dataset'
    ensemble_size = 5


resize_val = M.Compose([
    M.EnsureChannelFirst(channel_dim=-1),
    M.CropForeground(),
    M.Resize(CFG.img_size, size_mode='longest'),
    M.SpatialPad((CFG.img_size, CFG.img_size)),
    M.AsChannelLast(),
])

def get_transforms(data):
    if data == 'train':
        return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.PadIfNeeded(CFG.img_size, CFG.img_size),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            ToTensorV2(),
        ])

class GenderDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(CFG.data_dir, f"{row['id']}.png")
        img = np.array(Image.open(img_path).convert('L'))  # Grayscale

        # Create a 3-channel image
        img = np.stack([img, img, img], axis=-1)

        # Apply monai pipeline
        img = resize_val(img)
        img = img.astype(np.float32) / 255.

        if self.transforms:
            img = self.transforms(image=img)['image']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        label = int(row['male'])
        return img, torch.tensor(label, dtype=torch.float32)

class CustomModel(nn.Module):
    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, in_chans=3)
        if name in ['resnet50','resnet34','resnet18']:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 1)  # Single output, for sigmoid/BCE
    def forward(self, x, gender=None):
        x = self.model(x)
        return x

def train_fn(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(CFG.device), labels.to(CFG.device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def valid_fn(model, loader, criterion):
    model.eval()
    running_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device).unsqueeze(1)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = (np.array(all_preds).reshape(-1) == np.array(all_labels).reshape(-1)).mean()
    return running_loss / len(loader.dataset), acc

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Projetos\hand-bone\boneage-training-dataset.csv')
    # Ensure bools are interpreted correctly
    df['male'] = df['male'].astype(str).map({'True': True, 'False': False})

    for k in range(CFG.ensemble_size):
        print(f"\n=== Training model {k+1}/{CFG.ensemble_size} ===")
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=42+k, stratify=df['male']
        )

        train_set = GenderDataset(train_df, transforms=get_transforms('train'))
        val_set = GenderDataset(val_df, transforms=get_transforms('valid'))

        train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.n_workers)
        val_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.n_workers)

        model = CustomModel(name='resnet50', pretrained=True).to(CFG.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
        criterion = nn.BCEWithLogitsLoss()

        best_acc = 0
        for epoch in range(CFG.epochs):
            train_loss = train_fn(model, train_loader, optimizer, criterion)
            val_loss, val_acc = valid_fn(model, val_loader, criterion)
            print(f"Model {k+1} | Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"resnet50_gender_{k}.pth")
        print(f"Model {k+1} best validation accuracy:", best_acc)
