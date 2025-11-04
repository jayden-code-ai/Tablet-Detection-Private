#!/usr/bin/env python3
"""
Stage 7 (Ensemble): Validation GT vs YOLOv8l + YOLOv8x WBF 예측 시각화.

- 좌측: val.json 정답 박스
- 우측: YOLOv8l/yolov8x 예측을 Weighted Boxes Fusion으로 결합한 결과
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from ensemble_boxes import weighted_boxes_fusion
from matplotlib import font_manager, patches, rcParams
from PIL import Image
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path("/mnt/nas/jayden_code/Tablet-Detection-Private")
VAL_COCO_PATH = PROJECT_ROOT / "stage3_dataset_artifacts" / "coco" / "val.json"
CLASS_MAPPING_PATH = PROJECT_ROOT / "stage4_yolo_dataset" / "reports" / "class_mapping.json"
TRAIN_IMAGE_DIR = Path("/mnt/nas/jayden_code/ai05-level1-project/train_images")
DEFAULT_WEIGHTS = [
    PROJECT_ROOT / "stage5_yolov8l_runs" / "yolov8l_stage12" / "weights" / "best.pt",
    PROJECT_ROOT / "stage5_yolov8x_runs" / "yolov8x_stage12" / "weights" / "best.pt",
]
OUTPUT_DIR = PROJECT_ROOT / "stage7_visual_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COUNT = 6
DEFAULT_SEED = 2024
DEFAULT_IOU = 0.55
DEFAULT_SKIP = 0.001


@dataclass
class Annotation:
    category_id: int
    category_name: str
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)


def configure_korean_font() -> None:
    candidates = [
        "nanumsquareround",
        "nanumsquare",
        "nanumbarungothic",
        "nanumgothic",
        "notosanscjkkr",
        "notosanscjk",
        "notoserifcjkkr",
    ]
    system_fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font_path in system_fonts:
        lower = font_path.lower().replace(" ", "")
        if any(keyword in lower for keyword in candidates):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                continue

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for keyword in candidates:
        for font_name in available_fonts:
            if keyword in font_name.lower().replace(" ", ""):
                rcParams["font.family"] = font_name
                rcParams["axes.unicode_minus"] = False
                return
    rcParams["axes.unicode_minus"] = False


def load_coco_annotations(path: Path) -> Tuple[
    Dict[int, str],
    Dict[int, str],
    Dict[int, List[Annotation]],
]:
    data = json.loads(path.read_text(encoding="utf-8"))
    id_to_file: Dict[int, str] = {}
    id_to_cat: Dict[int, str] = {}
    ann_by_image: Dict[int, List[Annotation]] = {}

    for image in data.get("images", []):
        id_to_file[int(image["id"])] = image["file_name"]

    for cat in data.get("categories", []):
        id_to_cat[int(cat["id"])] = cat["name"]

    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        bbox = tuple(float(x) for x in ann["bbox"])
        ann_obj = Annotation(
            category_id=category_id,
            category_name=id_to_cat.get(category_id, str(category_id)),
            bbox=bbox,
        )
        ann_by_image.setdefault(image_id, []).append(ann_obj)

    return id_to_file, id_to_cat, ann_by_image


def load_inverse_mapping(path: Path) -> Dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload["class_mapping"]  # original cat id(str) -> index
    inverse: Dict[int, int] = {}
    for original_id_str, idx in mapping.items():
        inverse[int(idx)] = int(original_id_str)
    return inverse


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, x1, img_w - 1)
    y2 = clamp(y2, y1, img_h - 1)
    w = clamp(x2 - x1, 1, img_w - x1)
    h = clamp(y2 - y1, 1, img_h - y1)
    return x1, y1, w, h


def normalize(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (
        clamp(x / width, 0.0, 1.0),
        clamp(y / height, 0.0, 1.0),
        clamp((x + w) / width, 0.0, 1.0),
        clamp((y + h) / height, 0.0, 1.0),
    )


def denormalize(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = clamp(box[0], 0.0, 1.0) * width
    y1 = clamp(box[1], 0.0, 1.0) * height
    x2 = clamp(box[2], 0.0, 1.0) * width
    y2 = clamp(box[3], 0.0, 1.0) * height
    x = clamp(x1, 0.0, width - 1)
    y = clamp(y1, 0.0, height - 1)
    w = clamp(x2 - x1, 1.0, width - x)
    h = clamp(y2 - y1, 1.0, height - y)
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def draw_annotations(
    ax: plt.Axes,
    image: Image.Image,
    annotations: Iterable[Annotation],
    title: str,
    edge_color: str,
) -> None:
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    for ann in annotations:
        x, y, w, h = ann.bbox
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor=edge_color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y - 4,
            ann.category_name,
            color=edge_color,
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.4, "pad": 1},
        )


def run_ensemble_prediction(
    models: List[YOLO],
    image_path: Path,
    inverse_mapping: Dict[int, int],
    iou_thr: float,
    skip_box_thr: float,
) -> List[Annotation]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    width, height = Image.open(image_path).size

    boxes_list: List[List[Tuple[float, float, float, float]]] = []
    scores_list: List[List[float]] = []
    labels_list: List[List[int]] = []

    for model in models:
        result = model.predict(
            source=str(image_path),
            device=device,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        det_boxes = []
        det_scores = []
        det_labels = []
        for (x1, y1, x2, y2), cls_idx, score in zip(xyxy, cls, conf):
            bbox = xyxy_to_xywh(x1, y1, x2, y2, img_w=width, img_h=height)
            det_boxes.append(normalize(bbox, width, height))
            det_scores.append(float(score))
            det_labels.append(inverse_mapping.get(int(cls_idx), int(cls_idx)))

        boxes_list.append(det_boxes)
        scores_list.append(det_scores)
        labels_list.append(det_labels)

    if not any(scores_list):
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=[1.0] * len(models),
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    annotations: List[Annotation] = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x, y, w, h = denormalize(box, width, height)
        annotations.append(
            Annotation(
                category_id=int(label),
                category_name=str(int(label)),
                bbox=(x, y, w, h),
            )
        )
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT vs Ensemble prediction visualization")
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        default=DEFAULT_WEIGHTS,
        help="앙상블에 사용할 YOLO 가중치 경로 목록",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="랜덤 샘플 수",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="",
        help="콤마로 구분된 image_id 목록(지정 시 랜덤 추출 무시)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="랜덤 시드",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=DEFAULT_IOU,
        help="WBF IoU threshold",
    )
    parser.add_argument(
        "--skip-box-thr",
        type=float,
        default=DEFAULT_SKIP,
        help="WBF skip box threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_korean_font()

    if not VAL_COCO_PATH.exists():
        raise FileNotFoundError(f"val.json 을 찾을 수 없습니다: {VAL_COCO_PATH}")
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(f"class_mapping.json 을 찾을 수 없습니다: {CLASS_MAPPING_PATH}")

    id_to_file, id_to_cat, ann_by_image = load_coco_annotations(VAL_COCO_PATH)
    inverse_mapping = load_inverse_mapping(CLASS_MAPPING_PATH)

    if args.images:
        sample_ids = []
        for token in args.images.split(","):
            token = token.strip()
            if not token:
                continue
            image_id = int(token)
            if image_id not in id_to_file:
                raise ValueError(f"val.json에 존재하지 않는 image_id: {image_id}")
            sample_ids.append(image_id)
    else:
        image_ids = sorted(id_to_file.keys())
        rng = random.Random(args.seed)
        rng.shuffle(image_ids)
        sample_ids = image_ids[: max(0, min(args.count, len(image_ids)))]

    models = []
    for weight_path in args.weights:
        if not weight_path.exists():
            raise FileNotFoundError(f"가중치를 찾을 수 없습니다: {weight_path}")
        models.append(YOLO(str(weight_path)))

    for image_id in sample_ids:
        file_name = id_to_file[image_id]
        image_path = TRAIN_IMAGE_DIR / file_name
        if not image_path.exists():
            print(f"[WARN] 이미지가 존재하지 않아 건너뜁니다: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        gt_annotations = ann_by_image.get(image_id, [])
        pred_annotations = run_ensemble_prediction(
            models,
            image_path=image_path,
            inverse_mapping=inverse_mapping,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
        )

        # 카테고리 이름 매핑
        for ann in gt_annotations:
            ann.category_name = id_to_cat.get(ann.category_id, str(ann.category_id))
        for ann in pred_annotations:
            ann.category_name = id_to_cat.get(ann.category_id, str(ann.category_id))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        draw_annotations(axes[0], image, gt_annotations, f"GT: {image_id}", edge_color="#1f77b4")
        draw_annotations(axes[1], image, pred_annotations, "Ensemble Pred", edge_color="#d62728")

        fig.tight_layout()
        output_path = OUTPUT_DIR / f"ensemble_compare_{image_id}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"[INFO] 시각화 저장: {output_path}")


if __name__ == "__main__":
    main()
