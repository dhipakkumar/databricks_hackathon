import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")          # change if your data lives elsewhere
TRAIN_DIR  = DATA_DIR / "Training"
TEST_DIR   = DATA_DIR / "Testing"
MODEL_DIR = Path("/tmp/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES    = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# ── Dataset ──────────────────────────────────────────────────────────────────
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, cls in enumerate(CLASSES):
            cls_dir = Path(root_dir) / cls
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Feature extractor ────────────────────────────────────────────────────────
def build_extractor():
    """EfficientNetB0 with the classifier head removed — outputs 1280-d features."""
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Identity()   # strip the head
    backbone.eval()
    return backbone.to(DEVICE)


def extract_features(loader, model):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            feats = model(imgs).cpu().numpy()
            features.append(feats)
            labels.extend(lbls.numpy())
    return np.vstack(features), np.array(labels)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    print("Loading datasets …")
    train_ds = BrainTumorDataset(TRAIN_DIR, transform)
    test_ds  = BrainTumorDataset(TEST_DIR,  transform)
    print(f"  Train: {len(train_ds)} images  |  Test: {len(test_ds)} images")

    train_loader = DataLoader(train_ds,batch_size=16,num_workers=0,pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size = 16, num_workers = 0, pin_memory = False)
    # ── Feature extraction ───────────────────────────────────────────────────
    print(f"\nExtracting features with EfficientNetB0 on {DEVICE} …")
    extractor = build_extractor()

    X_train, y_train = extract_features(train_loader, extractor)
    X_test,  y_test  = extract_features(test_loader,  extractor)
    print(f"  Feature shape — train: {X_train.shape}  test: {X_test.shape}")

    np.save(MODEL_DIR / "X_train.npy", X_train)
    np.save(MODEL_DIR / "y_train.npy", y_train)
    np.save(MODEL_DIR / "X_test.npy",  X_test)
    np.save(MODEL_DIR / "y_test.npy",  y_test)

    # ── XGBoost training ─────────────────────────────────────────────────────
    print("\nTraining XGBoost classifier …")
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",          # fast on CPU; set to "gpu_hist" if CUDA available
        device=DEVICE if DEVICE == "cuda" else "cpu",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    joblib.dump(clf, MODEL_DIR / "xgb_brain_tumor.pkl")
    print("  Model saved to models/xgb_brain_tumor.pkl")

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix — Test Accuracy: {acc:.2%}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    print("  Confusion matrix saved to models/confusion_matrix.png")

    # Feature importance (top 20)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-20:]
    plt.figure(figsize=(8, 5))
    plt.barh(range(20), importances[top_idx])
    plt.yticks(range(20), [f"feat_{i}" for i in top_idx])
    plt.title("Top-20 Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
    print("  Feature importance plot saved to models/feature_importance.png")


if __name__ == "__main__":
    main()
