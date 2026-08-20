"""
Chạy trước train.py để kiểm tra nhanh path/cấu trúc dữ liệu, tránh tốn giờ Colab
nếu config sai. Không cần GPU.

Cách chạy (trên Colab, sau khi mount Drive):
    python check_setup.py
"""
import os
import pandas as pd
import cv2

from config import cfg
from dataset import load_split_csv, RSNABoneAgeDataset

def main():
    print(f"data_root   = {cfg.data_root}")
    print(f"output_dir  = {cfg.output_dir}")
    ok = True

    if not os.path.isdir(cfg.data_root):
        print(f"[LỖI] Không tồn tại thư mục data_root. Kiểm tra đã mount Drive chưa "
              f"(drive.mount('/content/drive')) và path đúng chưa.")
        return

    csv_path = os.path.join(cfg.data_root, cfg.train_csv)
    if not os.path.exists(csv_path):
        print(f"[LỖI] Không tìm thấy {csv_path}")
        return
    print(f"[OK] Tìm thấy {cfg.train_csv}")

    df = load_split_csv(cfg)
    print(f"[OK] Đọc CSV: {len(df)} dòng, cột: {list(df.columns)}")
    n_male = int((df["male"] == 1).sum())
    n_female = int((df["male"] == 0).sum())
    print(f"      Nam: {n_male} | Nữ: {n_female} "
          f"(RSNA gốc kỳ vọng ~6833 nam / ~5778 nữ trên 12611 ảnh train)")
    if len(df) < 1000:
        print("[CẢNH BÁO] Số dòng CSV quá ít so với RSNA gốc (12,611) — kiểm tra lại file CSV.")
        ok = False

    img_dir = os.path.join(cfg.data_root, cfg.train_img_dir)
    try:
        ds = RSNABoneAgeDataset(df.head(5), img_dir, cfg, train=False)
        sample = ds[0]
        print(f"[OK] Đọc thử ảnh thành công. Thư mục ảnh thực tế: {ds.img_dir}")
        print(f"      Tensor shape: {tuple(sample['image'].shape)} "
              f"(kỳ vọng: (3, {cfg.image_size}, {cfg.image_size}))")
    except Exception as e:
        print(f"[LỖI] Không đọc được ảnh mẫu: {e}")
        ok = False

    n_images = len([f for f in os.listdir(
        img_dir if os.path.isdir(img_dir) else cfg.data_root) if f.endswith(".png")])
    print(f"      Số file .png đếm được ở '{img_dir}': {n_images}")

    print("\n" + ("=== SETUP OK, có thể chạy train.py ===" if ok else
                   "=== CÒN LỖI, sửa trước khi chạy train.py ==="))


if __name__ == "__main__":
    main()
