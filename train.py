# Databricks Brain Tumor Classifier with Spark + Delta
# Works with Workspace files, not DBFS FileStore

import io
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import xgboost as xgb
import joblib

from sklearn.metrics import accuracy_score, classification_report

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit


spark = SparkSession.builder.getOrCreate()


# ── Config ─────────────────────────────────────────────

BASE_PATH = "/Workspace/Users/na24b007@smail.iitm.ac.in/submission/data"

LOCAL_MODEL_DIR = "/Workspace/Users/na24b007@smail.iitm.ac.in/submission/models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


CLASSES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Step 1: Bronze Delta Table ──────────────────────────

def create_bronze_table():
    import os

    # IMPORTANT: local path, not file:/ path
    LOCAL_BASE_PATH = "/Workspace/Users/na24b007@smail.iitm.ac.in/submission/data"

    rows = []

    for split in ["Training", "Testing"]:
        for label_id, cls in enumerate(CLASSES):
            folder = os.path.join(LOCAL_BASE_PATH, split, cls)

            print("Reading:", folder)

            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder, filename)

                    with open(img_path, "rb") as f:
                        content = f.read()

                    rows.append({
                        "path": img_path,
                        "content": bytearray(content),
                        "split": split,
                        "label_name": cls,
                        "label": label_id
                    })

    bronze_df = spark.createDataFrame(rows)

    bronze_df.write.format("delta").mode("overwrite").saveAsTable(
        "brain_tumor_bronze_images"
    )

    print("Bronze table created: brain_tumor_bronze_images")
    print("Total images:", bronze_df.count())
# ── Step 2: EfficientNet Feature Extractor ──────────────

def build_extractor():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier = nn.Identity()
    model.eval()
    return model.to(DEVICE)


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


def image_bytes_to_tensor(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0)


# ── Step 3: Silver Delta Table ──────────────────────────

def create_silver_features():
    extractor = build_extractor()

    bronze_df = spark.table("brain_tumor_bronze_images")

    rows = (
        bronze_df
        .select("path", "content", "split", "label_name", "label")
        .collect()
    )

    feature_rows = []

    print("Extracting features on:", DEVICE)

    with torch.no_grad():
        for i, row in enumerate(rows):
            img_tensor = image_bytes_to_tensor(row["content"]).to(DEVICE)

            features = extractor(img_tensor)
            features = features.cpu().numpy().flatten()

            feature_rows.append({
                "path": row["path"],
                "split": row["split"],
                "label_name": row["label_name"],
                "label": int(row["label"]),
                "features": features.tolist()
            })

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(rows)} images")

    features_pdf = pd.DataFrame(feature_rows)

    silver_df = spark.createDataFrame(features_pdf)

    silver_df.write.format("delta").mode("overwrite").saveAsTable(
        "brain_tumor_silver_features"
    )

    print("Silver table created: brain_tumor_silver_features")


# ── Step 4: Train XGBoost + Gold Delta Tables ───────────

def train_model():
    silver_df = spark.table("brain_tumor_silver_features")
    pdf = silver_df.toPandas()

    train_pdf = pdf[pdf["split"] == "Training"]
    test_pdf = pdf[pdf["split"] == "Testing"]

    X_train = np.array(train_pdf["features"].tolist())
    y_train = train_pdf["label"].values

    X_test = np.array(test_pdf["features"].tolist())
    y_test = test_pdf["label"].values

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    model_path = os.path.join(
        LOCAL_MODEL_DIR,
        "xgb_brain_tumor_delta.pkl"
    )

    joblib.dump(clf, model_path)

    print("Model saved to:")
    print(model_path)

    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    pred_rows = []

    for i in range(len(test_pdf)):
        pred_rows.append({
            "path": test_pdf.iloc[i]["path"],
            "true_label": int(y_test[i]),
            "true_class": CLASSES[int(y_test[i])],
            "predicted_label": int(y_pred[i]),
            "predicted_class": CLASSES[int(y_pred[i])],
            "confidence": float(np.max(probs[i])),
            "correct": bool(y_test[i] == y_pred[i])
        })

    pred_pdf = pd.DataFrame(pred_rows)

    spark.createDataFrame(pred_pdf) \
        .write.format("delta") \
        .mode("overwrite") \
        .saveAsTable("brain_tumor_gold_predictions")

    metrics_pdf = pd.DataFrame([{
        "accuracy": float(acc),
        "model": "EfficientNetB0 + XGBoost",
        "model_path": model_path,
        "bronze_table": "brain_tumor_bronze_images",
        "silver_table": "brain_tumor_silver_features",
        "gold_table": "brain_tumor_gold_predictions"
    }])

    spark.createDataFrame(metrics_pdf) \
        .write.format("delta") \
        .mode("overwrite") \
        .saveAsTable("brain_tumor_gold_metrics")

    print("Gold tables created:")
    print("brain_tumor_gold_predictions")
    print("brain_tumor_gold_metrics")


# ── Run Pipeline ────────────────────────────────────────

create_bronze_table()
create_silver_features()
train_model()
