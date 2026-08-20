"""
Entry point Giai đoạn 1.

Quy trình (khớp baseline + thay đổi Giai đoạn 1):
1. Với mỗi giới tính (nếu sex_specific=True) và mỗi kiến trúc trong cfg.architectures:
   train 5-fold CV, lưu checkpoint + out-of-fold (OOF) predictions.
2. Fit ArchitectureEnsemble (stacking) trên OOF predictions của 3 kiến trúc.
3. Đánh giá trên test set RSNA gốc (200 ảnh), so với baseline MAE 6.26 / RMSE 7.79.

Baseline gốc: MAE 6.26 tháng (95% CI 5.60–6.89), RMSE 7.79 tháng (95% CI 7.02–8.65).
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from config import cfg, grad_accum_for_arch
from dataset import RSNABoneAgeDataset, load_split_csv
from models import build_model
from ensemble import ArchitectureEnsemble


def get_loss_fn(cfg):
    if cfg.loss == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(cfg.loss)


def train_one_fold(arch, train_df, val_df, cfg, fold_idx, sex_tag):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(arch, cfg).to(device)

    phys_bs = cfg.per_arch_batch_size.get(arch, cfg.batch_size)
    accum_steps = grad_accum_for_arch(cfg, arch)  # giữ effective batch size ~ khớp baseline

    train_ds = RSNABoneAgeDataset(train_df, os.path.join(cfg.data_root, cfg.train_img_dir), cfg, train=True)
    val_ds = RSNABoneAgeDataset(val_df, os.path.join(cfg.data_root, cfg.train_img_dir), cfg, train=False)
    train_dl = DataLoader(train_ds, batch_size=phys_bs, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=phys_bs, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)

    loss_fn = get_loss_fn(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_ckpt = os.path.join(cfg.output_dir, f"{arch}_{sex_tag}_fold{fold_idx}_best.pt")
    last_ckpt = os.path.join(cfg.output_dir, f"{arch}_{sex_tag}_fold{fold_idx}_last.pt")

    start_epoch = 0
    best_mae = float("inf")

    # == Resume: nếu bị ngắt (Colab hết giờ/hết GPU), tiếp tục từ epoch cuối đã lưu ==
    if cfg.resume and os.path.exists(last_ckpt):
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_mae = state["best_mae"]
        print(f"[{arch}|{sex_tag}|fold{fold_idx}] resume từ epoch {start_epoch} "
              f"(best_mae hiện tại = {best_mae:.3f})")

    if start_epoch >= cfg.epochs:
        print(f"[{arch}|{sex_tag}|fold{fold_idx}] đã train xong trước đó, bỏ qua.")
        return best_ckpt, best_mae

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_dl):
            image = batch["image"].to(device, non_blocking=True)
            gender = batch["gender"].to(device, non_blocking=True)
            target = batch["boneage"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                pred = model(image, gender)
                loss = loss_fn(pred, target) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        scheduler.step()

        val_mae, _ = evaluate(model, val_dl, device)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_ckpt)

        # Lưu checkpoint "last" mỗi N epoch để resume nếu Colab ngắt giữa chừng
        if (epoch + 1) % cfg.save_every_n_epochs == 0 or epoch == cfg.epochs - 1:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_mae": best_mae,
            }, last_ckpt)

        print(f"[{arch}|{sex_tag}|fold{fold_idx}] epoch {epoch+1}/{cfg.epochs} "
              f"val_mae={val_mae:.3f} best={best_mae:.3f}")

    print(f"[{arch}|{sex_tag}|fold{fold_idx}] best val MAE = {best_mae:.3f} tháng -> {best_ckpt}")
    return best_ckpt, best_mae


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    preds, targets, ids = [], [], []
    for batch in dl:
        image = batch["image"].to(device)
        gender = batch["gender"].to(device)
        pred = model(image, gender)
        preds.append(pred.cpu().numpy())
        targets.append(batch["boneage"].numpy())
        ids.append(batch["id"].numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    ids = np.concatenate(ids)
    mae = mean_absolute_error(targets, preds)
    return mae, (ids, preds, targets)


def run_phase1(cfg):
    df = load_split_csv(cfg)
    sex_groups = [(1, "male"), (0, "female")] if cfg.sex_specific else [(None, "all")]

    oof_path = os.path.join(cfg.output_dir, "oof_predictions.csv")
    progress_path = os.path.join(cfg.output_dir, "progress.json")

    # == Resume ở cấp toàn cục: bỏ qua (arch, sex_tag, fold) đã train xong từ lần chạy trước ==
    oof_records = []
    done_keys = set()
    if os.path.exists(oof_path):
        oof_df_prev = pd.read_csv(oof_path)
        oof_records = oof_df_prev.to_dict("records")
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            done_keys = set(json.load(f))
        print(f"Đã tìm thấy progress.json: {len(done_keys)} fold đã hoàn thành, sẽ bỏ qua.")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    for sex_val, sex_tag in sex_groups:
        sub_df = df if sex_val is None else df[df["male"] == sex_val].reset_index(drop=True)
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

        for arch in cfg.architectures:
            for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(sub_df)):
                key = f"{arch}|{sex_tag}|fold{fold_idx}"
                train_df = sub_df.iloc[tr_idx].reset_index(drop=True)
                val_df = sub_df.iloc[va_idx].reset_index(drop=True)

                if key in done_keys:
                    print(f"[skip] {key} đã hoàn thành trước đó.")
                    continue

                ckpt_path, _ = train_one_fold(arch, train_df, val_df, cfg, fold_idx, sex_tag)

                model = build_model(arch, cfg).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                phys_bs = cfg.per_arch_batch_size.get(arch, cfg.batch_size)
                val_ds = RSNABoneAgeDataset(val_df, os.path.join(cfg.data_root, cfg.train_img_dir), cfg, train=False)
                val_dl = DataLoader(val_ds, batch_size=phys_bs, shuffle=False)
                _, (ids, preds, targets) = evaluate(model, val_dl, device)

                for i, p, t in zip(ids, preds, targets):
                    oof_records.append({"id": int(i), "arch": arch, "sex_tag": sex_tag,
                                         "pred": float(p), "target": float(t)})

                # Lưu ngay sau mỗi fold hoàn thành — nếu Colab ngắt giữa các fold, không mất tiến độ
                done_keys.add(key)
                pd.DataFrame(oof_records).to_csv(oof_path, index=False)
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump(sorted(done_keys), f, ensure_ascii=False, indent=2)

    oof_df = pd.DataFrame(oof_records)
    print(f"Đã lưu OOF predictions -> {oof_path}")

    # Fit stacking ensemble trên OOF (mỗi id chỉ có 1 pred/arch vì mỗi id chỉ ở 1 fold)
    pivot = oof_df.pivot_table(index="id", columns="arch", values="pred", aggfunc="first")
    y_true = oof_df.drop_duplicates("id").set_index("id")["target"].loc[pivot.index]

    ensemble = ArchitectureEnsemble(mode=cfg.ensemble_mode)
    oof_preds_dict = {a: pivot[a].values for a in pivot.columns}
    ensemble.fit(oof_preds_dict, y_true.values)
    final_oof_pred = ensemble.predict(oof_preds_dict)

    mae = mean_absolute_error(y_true.values, final_oof_pred)
    rmse = np.sqrt(mean_squared_error(y_true.values, final_oof_pred))

    report = {
        "mae_months": round(float(mae), 3),
        "rmse_months": round(float(rmse), 3),
        "baseline_mae_months": 6.26,
        "baseline_rmse_months": 7.79,
        "delta_mae": round(float(mae) - 6.26, 3),
        "ensemble_weights": ensemble.weights_report(),
        "architectures": list(cfg.architectures),
    }
    with open(os.path.join(cfg.output_dir, "phase1_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run_phase1(cfg)
