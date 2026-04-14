"""
Brand Monitor — src/detector.py
==================================
Image-based counterfeit detection using:
  - YOLOv8    (object detection + classification)
  - EfficientNet-B3 (4-class image classifier)
  - BERT + TF-IDF LR ensemble (text classification)

Model files expected in /models/:
  best.pt                  — YOLOv8 weights
  efficientnet_4class.pth  — EfficientNet checkpoint
  bert_brand.pth           — BERT checkpoint
  tfidf_vectorizer.pkl     — TF-IDF vectorizer
  lr_model.pkl             — Logistic Regression model
"""

import io
import pickle
import numpy as np
import torch
import torch.nn as nn

from PIL          import Image
from torchvision  import transforms, models
from transformers import BertTokenizer, BertForSequenceClassification
from ultralytics  import YOLO

# ══════════════════════════════════════════════════════════════════
# CLASS MAPS
# Edit these to match your model's training labels
# ══════════════════════════════════════════════════════════════════

YOLO_FAKE_IDS = {0, 1}   # class IDs that mean "fake"
YOLO_REAL_IDS = {2, 3}   # class IDs that mean "real"

YOLO_NAMES = {
    0: "fake_variant_a",
    1: "fake_variant_b",
    2: "real_variant_a",
    3: "real_variant_b"
}

EFF_CLASSES = [
    "fake_variant_a",
    "fake_variant_b",
    "real_variant_a",
    "real_variant_b"
]

# ══════════════════════════════════════════════════════════════════
# IMAGE TRANSFORM (must match EfficientNet training)
# ══════════════════════════════════════════════════════════════════

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ══════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════

def load_yolo(path: str = "models/best.pt") -> YOLO:
    return YOLO(path)


def load_efficientnet(path: str = "models/efficientnet_4class.pth"):
    model = models.efficientnet_b3(pretrained=False)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(EFF_CLASSES)
    )
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_bert(path: str = "models/bert_brand.pth"):
    """Load BERT classifier. Returns (model, tokenizer, classes_list)."""
    checkpoint = torch.load(path, map_location="cpu")
    classes    = checkpoint["classes"]
    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    model      = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(classes)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, classes


def load_tfidf_lr(
    tfidf_path: str = "models/tfidf_vectorizer.pkl",
    lr_path:    str = "models/lr_model.pkl"
):
    """Load TF-IDF vectorizer + Logistic Regression. Returns (tfidf, lr)."""
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(lr_path, "rb") as f:
        lr = pickle.load(f)
    return tfidf, lr


# ══════════════════════════════════════════════════════════════════
# IMAGE COUNTERFEIT DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_counterfeit(
    image_bytes: bytes,
    yolo_model,
    eff_model
) -> dict:
    """
    Run YOLOv8 + EfficientNet on image bytes.
    Returns a combined verdict dict.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── YOLOv8 ───────────────────────────────────────────────────
    yolo_results = yolo_model(image)
    detections   = []

    for r in yolo_results:
        for box in r.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            is_fake = cls_id in YOLO_FAKE_IDS
            detections.append({
                "label":      "counterfeit" if is_fake else "authentic",
                "class_name": YOLO_NAMES.get(cls_id, "unknown"),
                "confidence": round(conf * 100, 2),
                "box":        box.xyxy[0].tolist()
            })

    if detections:
        best         = max(detections, key=lambda x: x["confidence"])
        yolo_verdict = best["label"]
        yolo_conf    = best["confidence"]
        yolo_name    = best["class_name"]
    else:
        yolo_verdict = "authentic"
        yolo_conf    = 40.0
        yolo_name    = "no detection"

    # ── EfficientNet ──────────────────────────────────────────────
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = eff_model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)

    class_scores = {
        EFF_CLASSES[i]: round(float(probs[0][i].item()) * 100, 2)
        for i in range(len(EFF_CLASSES))
    }

    fake_score = sum(
        class_scores[c] for c in EFF_CLASSES if c.startswith("fake")
    )
    real_score = sum(
        class_scores[c] for c in EFF_CLASSES if c.startswith("real")
    )

    eff_verdict = "counterfeit" if fake_score > real_score else "authentic"
    eff_conf    = fake_score if eff_verdict == "counterfeit" else real_score

    eff_result = {
        "label":        eff_verdict,
        "confidence":   round(eff_conf, 2),
        "class_scores": class_scores,
        "fake_score":   round(fake_score, 2),
        "real_score":   round(real_score, 2)
    }

    # ── Ensemble ──────────────────────────────────────────────────
    yolo_cf  = yolo_conf if yolo_verdict == "counterfeit" else (100 - yolo_conf)
    eff_cf   = eff_conf  if eff_verdict  == "counterfeit" else (100 - eff_conf)
    combined = (yolo_cf * 0.45) + (eff_cf * 0.55)

    # Agreement → more decisive; disagreement → use EfficientNet (higher weight)
    if yolo_verdict == "counterfeit" and eff_verdict == "counterfeit":
        verdict    = "COUNTERFEIT"
        risk_score = round(combined, 2)
    elif yolo_verdict == "authentic" and eff_verdict == "authentic":
        verdict    = "AUTHENTIC"
        risk_score = round(100 - combined, 2)
    else:
        # Disagreement: trust EfficientNet
        verdict    = "COUNTERFEIT" if eff_verdict == "counterfeit" else "AUTHENTIC"
        risk_score = round(combined, 2)

    return {
        "yolo":            detections,
        "yolo_name":       yolo_name,
        "efficientnet":    eff_result,
        "risk_score":      risk_score,
        "verdict":         verdict,
        "model_agreement": yolo_verdict == eff_verdict
    }


# ══════════════════════════════════════════════════════════════════
# TEXT SENTIMENT + ENSEMBLE
# ══════════════════════════════════════════════════════════════════

def analyze_sentiment(
    text:         str,
    bert_model,
    tokenizer,
    classes:      list,
    tfidf=None,
    lr=None,
    bert_weight:  float = 0.6,
    lr_weight:    float = 0.4
) -> dict:
    """
    Classify text using BERT (+ optional TF-IDF LR ensemble).
    Returns label, confidence, and per-class scores.
    """
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs       = bert_model(**encoding)
        bert_probs    = torch.softmax(outputs.logits, dim=1)
        bert_probs_np = bert_probs.numpy()[0]

    bert_pred_idx  = int(bert_probs_np.argmax())
    bert_pred_conf = float(bert_probs_np[bert_pred_idx]) * 100

    # ── TF-IDF + LR ensemble (optional) ──────────────────────────
    if tfidf is not None and lr is not None:
        try:
            text_vec   = tfidf.transform([text])
            lr_probs   = lr.predict_proba(text_vec)[0]
            lr_classes = lr.classes_
            lr_prob_map = {c: p for c, p in zip(lr_classes, lr_probs)}

            ensemble_probs = bert_probs_np.copy() * bert_weight

            for i, cls in enumerate(classes):
                if cls == "counterfeit_alert":
                    ensemble_probs[i] += lr_prob_map.get("fake", 0) * lr_weight
                elif cls == "positive":
                    ensemble_probs[i] += lr_prob_map.get("real", 0) * lr_weight

            # Normalise to sum to 1
            ensemble_probs = ensemble_probs / ensemble_probs.sum()

            final_idx   = int(ensemble_probs.argmax())
            final_label = classes[final_idx]
            final_conf  = round(float(ensemble_probs[final_idx]) * 100, 2)

            return {
                "label":            final_label,
                "confidence":       final_conf,
                "all_scores":       {
                    classes[i]: round(float(ensemble_probs[i]) * 100, 2)
                    for i in range(len(classes))
                },
                "bert_prediction":  classes[bert_pred_idx],
                "bert_confidence":  round(bert_pred_conf, 2),
                "lr_prediction":    "fake" if lr_prob_map.get("fake", 0) > 0.5 else "real",
                "lr_confidence":    round(max(lr_probs) * 100, 2),
                "ensemble_used":    True
            }
        except Exception as e:
            print(f"LR ensemble error ({e}) — falling back to BERT only")

    # ── BERT only fallback ────────────────────────────────────────
    return {
        "label":          classes[bert_pred_idx],
        "confidence":     round(bert_pred_conf, 2),
        "all_scores":     {
            classes[i]: round(float(bert_probs_np[i]) * 100, 2)
            for i in range(len(classes))
        },
        "ensemble_used":  False
    }