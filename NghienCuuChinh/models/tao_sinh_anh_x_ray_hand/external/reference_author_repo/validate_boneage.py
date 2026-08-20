import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

from app import predict

DATASET_DIR = r"C:\Projetos\hand-bone\output_cleaned_mult"
CSV_PATH = r"C:\Projetos\hand-bone\boneage-test-dataset.csv"
ENSEMBLE_DIR = r"C:\Projetos\hand-bone\bone_age_ensemble"
OUTPUT_CSV = r"C:\Projetos\hand-bone\cleaned_boneage_predictions_mult.csv"

# Leitura do CSV
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
df["Sex"] = df["Sex"].str.upper().str.strip()

results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    case_id = str(row["Case ID"])
    gender = row["Sex"]
    patient_dir = os.path.join(DATASET_DIR, case_id)

    if not os.path.exists(patient_dir):
        print(f"[WARNING] Directory not found for patient: {patient_dir}")
        continue

    # Find all images matching the pattern: {case_id}_cleaned_*.png
    img_files = [f for f in os.listdir(patient_dir) if f.startswith(f"{case_id}_cleaned_") and f.endswith(".png")]
    if not img_files:
        print(f"[WARNING] No images found for patient: {case_id}")
        continue

    for img_file in img_files:
        img_path = os.path.join(patient_dir, img_file)
        try:
            img = np.array(Image.open(img_path).convert("L"))
            pred_months = predict(img, gender, ENSEMBLE_DIR)
            results.append({
                "Case ID": case_id,
                "Sex": gender,
                "Image Name": img_file,
                "Predicted Months": pred_months
            })
        except Exception as e:
            print(f"[ERROR] Failed on {img_path}: {e}")

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to: {OUTPUT_CSV}")
