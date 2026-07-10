# ============================================
# AeroSentinel AI - Complete Backend
# ============================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import shutil, os, cv2, numpy as np

app = FastAPI(title="AeroSentinel AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ──────────────────────────────────────────────────────────────────
model = YOLO("best.pt")
CLASS_NAMES = model.names   # {0:'A220', 1:'A320-321', ...}

# ── Folders ────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DATASET_ROOT = r"C:\Users\pravalika ranka\Downloads\SAR_AIRCRAFT_1.0"
LABEL_SEARCH_DIRS = [
    os.path.join(DATASET_ROOT, "test",  "labels"),
    os.path.join(DATASET_ROOT, "valid", "labels"),
    os.path.join(DATASET_ROOT, "train", "labels"),
]

app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

# ── Session state ──────────────────────────────────────────────────────────
# Each entry: {"gt": cls_id, "pred": cls_id}   (both are valid class indices)
# Unmatched GT   → {"gt": cls_id, "pred": -1}  (false negative)
# Unmatched Pred → {"gt": -1,     "pred": cls_id} (false positive)
detection_log = []


# ── Auto GT lookup ─────────────────────────────────────────────────────────
def find_gt_label(image_filename: str) -> list:
    stem = os.path.splitext(image_filename)[0]
    for label_dir in LABEL_SEARCH_DIRS:
        label_path = os.path.join(label_dir, stem + ".txt")
        if os.path.exists(label_path):
            boxes = []
            with open(label_path) as f:
                for line in f.read().strip().splitlines():
                    parts = list(map(float, line.split()))
                    if len(parts) == 5:
                        boxes.append(parts)
            return boxes
    return []


# ── IoU ────────────────────────────────────────────────────────────────────
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ── Match GT boxes to predictions using greedy IoU ────────────────────────
def match_gt_pred(gt_boxes_norm, pred_boxes, img_w, img_h, iou_thresh=0.45):
    """
    Returns a list of dicts, one per GT box and one per unmatched Pred box:
      - Matched pair  : {"gt": gt_cls,  "pred": pred_cls}  ← IoU ≥ thresh
      - Unmatched GT  : {"gt": gt_cls,  "pred": -1}        ← False Negative
      - Unmatched Pred: {"gt": -1,      "pred": pred_cls}  ← False Positive

    This ensures the confusion matrix and FP/FN are accounted for correctly.
    """
    pairs     = []
    used_preds = set()

    # Convert GT normalised cx,cy,w,h → pixel xyxy
    gt_xyxy = []
    for box in gt_boxes_norm:
        cls_id = int(box[0])
        cx, cy, bw, bh = box[1], box[2], box[3], box[4]
        x1 = (cx - bw / 2) * img_w;  y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w;  y2 = (cy + bh / 2) * img_h
        gt_xyxy.append((cls_id, [x1, y1, x2, y2]))

    for gt_cls, gt_box in gt_xyxy:
        best_iou, best_idx = 0.0, -1
        for pi, pred in enumerate(pred_boxes):
            if pi in used_preds:
                continue
            iou = compute_iou(gt_box, pred["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, pi

        if best_iou >= iou_thresh and best_idx >= 0:
            # True match (could be TP if same class, or misclassification)
            pairs.append({
                "gt":   gt_cls,
                "pred": int(pred_boxes[best_idx]["class_id"])
            })
            used_preds.add(best_idx)
        else:
            # Unmatched GT → False Negative
            pairs.append({"gt": gt_cls, "pred": -1})

    # Remaining unmatched predictions → False Positives
    for pi, pred in enumerate(pred_boxes):
        if pi not in used_preds:
            pairs.append({"gt": -1, "pred": int(pred["class_id"])})

    return pairs


# ── Compute accurate metrics from detection_log ───────────────────────────
def compute_live_metrics():
    """
    Metrics are computed per-class from the full session log then macro-averaged.

    For each class c:
      TP_c = matched pairs where gt==c AND pred==c           (correct classification)
      FP_c = any pair where pred==c but gt != c              (wrong pred or unmatched pred)
      FN_c = any pair where gt==c  but pred != c             (missed or wrong pred)
      TN_c = everything else (not predicted as c AND not actually c)

    Precision_c = TP_c / (TP_c + FP_c)   — how reliable are predictions of class c
    Recall_c    = TP_c / (TP_c + FN_c)   — how well we catch all instances of class c
    Accuracy    = total_TP / total_matched_pairs  (matched GT↔Pred only, excludes unmatched)
    mAP@0.5     = mean of per-class AP proxied as Precision_c (at fixed IoU=0.45 threshold)

    Confusion matrix (n×n):
      row = predicted class, col = GT class
      Only matched pairs (both gt>=0 AND pred>=0) contribute.
    """
    n          = len(CLASS_NAMES)
    class_list = [CLASS_NAMES[i] for i in range(n)]

    # ── Per-class TP, FP, FN ──────────────────────────────────────────────
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n

    # n×n confusion matrix  row=predicted  col=actual(GT)
    matrix = [[0] * n for _ in range(n)]

    for entry in detection_log:
        gt   = entry["gt"]
        pred = entry["pred"]
        gt_valid   = (0 <= gt   < n)
        pred_valid = (0 <= pred < n)

        if gt_valid and pred_valid:
            # Matched pair — goes into confusion matrix
            matrix[pred][gt] += 1
            if pred == gt:
                tp[gt] += 1          # True Positive for this class
            else:
                fn[gt]   += 1        # GT class missed → FN for gt class
                fp[pred] += 1        # Wrong prediction → FP for pred class

        elif gt_valid and not pred_valid:
            # Unmatched GT (FN for that class)
            fn[gt] += 1

        elif pred_valid and not gt_valid:
            # Unmatched Pred (FP for that class)
            fp[pred] += 1

    # ── Metrics ──────────────────────────────────────────────────────────
    # Accuracy: correct matched pairs / all matched pairs (both sides valid)
    total_matched = sum(matrix[r][c] for r in range(n) for c in range(n))
    correct       = sum(matrix[i][i] for i in range(n))
    accuracy      = correct / total_matched if total_matched > 0 else 0.0

    # Macro precision, recall, and per-class AP (for mAP)
    per_class_precision = []
    per_class_recall    = []
    per_class_ap        = []   # AP proxy at IoU=0.45

    for c in range(n):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        # Only include classes that actually appear in GT or predictions
        if (tp[c] + fp[c] + fn[c]) > 0:
            per_class_precision.append(prec)
            per_class_recall.append(rec)
            # AP proxy: area under precision-recall curve simplified to prec*rec
            # (single-point approximation at our IoU threshold)
            ap = prec * rec  # bounded [0,1], consistent with single-threshold AP
            per_class_ap.append(ap)

    precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
    recall    = float(np.mean(per_class_recall))    if per_class_recall    else 0.0
    map50     = float(np.mean(per_class_ap))        if per_class_ap        else 0.0

    return {
        "precision":        round(precision,     4),
        "recall":           round(recall,        4),
        "accuracy":         round(accuracy,      4),
        "map50":            round(map50,         4),
        "class_names":      class_list,
        "confusion_matrix": matrix,
        "total_matched":    total_matched,
    }


# ── Draw GT (green) + Pred (red) boxes ────────────────────────────────────
def draw_boxes(img, gt_boxes, pred_boxes, class_names):
    h, w = img.shape[:2]

    for box in gt_boxes:
        cls_id = int(box[0])
        cx, cy, bw, bh = box[1], box[2], box[3], box[4]
        x1 = int((cx - bw / 2) * w);  y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w);  y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"GT:{class_names.get(cls_id, str(cls_id))}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        yy = max(y1 - 4, th + 6)
        cv2.rectangle(img, (x1, yy - th - 4), (x1 + tw + 4, yy + 2), (0, 160, 0), -1)
        cv2.putText(img, label, (x1 + 2, yy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)

    for det in pred_boxes:
        cls_id = int(det["class_id"]); conf = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Pred:{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        yy = min(y2 + th + 6, h - 2)
        cv2.rectangle(img, (x1, yy - th - 4), (x1 + tw + 4, yy + 2), (160, 0, 0), -1)
        cv2.putText(img, label, (x1 + 2, yy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    # Legend overlay
    cv2.rectangle(img, (8, 8), (250, 64), (0, 0, 0), -1)
    cv2.rectangle(img, (8, 8), (250, 64), (60, 60, 60),  1)
    cv2.putText(img, "GREEN = Ground Truth", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)
    cv2.putText(img, "RED   = Prediction",   (14, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1)
    return img


# ── Home ───────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "AeroSentinel AI Backend Running 🚀"}


# ── Clear session ──────────────────────────────────────────────────────────
@app.post("/clear-session")
def clear_session():
    global detection_log
    detection_log = []
    return {"message": "Session cleared"}


# ── Single Detection ───────────────────────────────────────────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global detection_log
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        results      = model(file_path, conf=0.25)
        img_cv       = cv2.imread(file_path)
        img_h, img_w = img_cv.shape[:2]

        pred_boxes = []
        for box in results[0].boxes:
            pred_boxes.append({
                "class_id":   int(box.cls[0]),
                "class_name": CLASS_NAMES.get(int(box.cls[0]), "unknown"),
                "confidence": float(box.conf[0]),
                "bbox":       [float(v) for v in box.xyxy[0]]
            })

        gt_boxes  = find_gt_label(file.filename)
        gt_source = "auto" if gt_boxes else "none"

        # Full matching: matched pairs + unmatched GT + unmatched preds
        pairs = match_gt_pred(gt_boxes, pred_boxes, img_w, img_h)
        detection_log.extend(pairs)

        img_cv = draw_boxes(img_cv, gt_boxes, pred_boxes, CLASS_NAMES)
        out_path = os.path.join(OUTPUT_FOLDER, file.filename)
        cv2.imwrite(out_path, img_cv)

        return JSONResponse(content={
            "detections": pred_boxes,
            "gt_count":   len(gt_boxes),
            "gt_source":  gt_source,
            "image_url":  f"/outputs/{file.filename}",
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── Batch Detection ────────────────────────────────────────────────────────
@app.post("/detect-batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    global detection_log
    results_list = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        results      = model(file_path, conf=0.25)
        gt_boxes     = find_gt_label(file.filename)
        img_cv       = cv2.imread(file_path)
        img_h, img_w = img_cv.shape[:2]

        pred_boxes = []
        for box in results[0].boxes:
            pred_boxes.append({
                "class_id":   int(box.cls[0]),
                "class_name": CLASS_NAMES.get(int(box.cls[0]), "unknown"),
                "confidence": float(box.conf[0]),
                "bbox":       [float(v) for v in box.xyxy[0]]
            })

        # Full matching: matched pairs + unmatched GT + unmatched preds
        pairs = match_gt_pred(gt_boxes, pred_boxes, img_w, img_h)
        detection_log.extend(pairs)

        img_cv = draw_boxes(img_cv, gt_boxes, pred_boxes, CLASS_NAMES)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, file.filename), img_cv)

        results_list.append({
            "filename":   file.filename,
            "image":      f"/outputs/{file.filename}",
            "detections": pred_boxes,
            "gt_count":   len(gt_boxes)
        })

    return {"results": results_list}


# ── Metrics endpoint ───────────────────────────────────────────────────────
@app.get("/metrics")
def get_metrics():
    return compute_live_metrics()