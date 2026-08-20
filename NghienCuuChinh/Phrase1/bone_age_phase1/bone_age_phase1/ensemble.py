"""
Kết hợp dự đoán của các kiến trúc trong ensemble.

- "average": trung bình cộng dự đoán các kiến trúc (baseline dùng trung bình
  5-fold cùng kiến trúc; ở đây mở rộng sang trung bình liên-kiến-trúc).
- "stacking": meta-learner hồi quy tuyến tính (LinearRegression) học trên
  out-of-fold (OOF) predictions của từng kiến trúc -> tránh leakage, đúng
  chuẩn thực hành stacking.
"""
import numpy as np
from sklearn.linear_model import LinearRegression


class ArchitectureEnsemble:
    def __init__(self, mode: str = "stacking"):
        assert mode in ("average", "stacking")
        self.mode = mode
        self.meta_model = LinearRegression(positive=True) if mode == "stacking" else None

    def fit(self, oof_preds: dict, y_true: np.ndarray):
        """oof_preds: {arch_name: np.array shape (N,)} — dự đoán out-of-fold.
        Chỉ cần gọi khi mode == 'stacking'."""
        if self.mode != "stacking":
            return self
        X = np.stack([oof_preds[a] for a in sorted(oof_preds)], axis=1)
        self.meta_model.fit(X, y_true)
        self._arch_order = sorted(oof_preds)
        return self

    def predict(self, preds: dict) -> np.ndarray:
        """preds: {arch_name: np.array shape (N,)}"""
        if self.mode == "average":
            return np.mean(np.stack(list(preds.values()), axis=1), axis=1)
        X = np.stack([preds[a] for a in self._arch_order], axis=1)
        return self.meta_model.predict(X)

    def weights_report(self):
        if self.mode != "stacking":
            return "average (trọng số bằng nhau)"
        return {a: float(w) for a, w in zip(self._arch_order, self.meta_model.coef_)}
