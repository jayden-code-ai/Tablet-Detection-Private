#!/usr/bin/env python3
"""
Stage 7: GT vs Prediction 시각화 생성 스크립트.

- 좌우 비교: 왼쪽은 원래 어노테이션(정답), 오른쪽은 YOLO 예측 결과
- 출력: /mnt/nas/jayden_code/Tablet-Detection-Private/stage7_visual_artifacts/compare_{image_id}.png
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager, patches, rcParams
from PIL import Image
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path("/mnt/nas/jayden_code/Tablet-Detection-Private")
VAL_COCO_PATH = PROJECT_ROOT / "stage3_dataset_artifacts" / "coco" / "val.json"
CLASS_MAPPING_PATH = PROJECT_ROOT / "stage4_yolo_dataset" / "reports" / "class_mapping.json"
TRAIN_IMAGE_DIR = Path("/mnt/nas/jayden_code/ai05-level1-project/train_images")
WEIGHTS_PATH = PROJECT_ROOT / "stage5_yolov8l_runs" / "yolov8l_stage12" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "stage7_visual_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_COUNT = 6
RANDOM_SEED = 2024


@dataclass
class Annotation:
    category_id: int
    category_name: str
    bbox: Tuple[float, float, float, float]


def configure_korean_font() -> None:
    candidate_keywords = [
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
        if any(keyword in lower for keyword in candidate_keywords):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                continue

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for keyword in candidate_keywords:
        for font_name in available_fonts:
            if keyword in font_name.lower().replace(" ", ""):
                rcParams["font.family"] = font_name
                rcParams["axes.unicode_minus"] = False
                return
    rcParams["axes.unicode_minus"] = False


def load_coco_annotations(path: Path) -> Tuple[
    Dict[int, str], Dict[int, str], Dict[int, List[Annotation]]
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


def draw_annotations(
    ax: plt.Axes,
    image: Image.Image,
    annotations: Iterable[Annotation],
    title: str,
    edge_color: str = "#1f77b4",
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


def load_inverse_mapping(path: Path) -> Dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload["class_mapping"]
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


def generate_visualizations(sample_ids: List[int], inverse_mapping: Dict[int, int]) -> None:
    configure_korean_font()
    model = YOLO(str(WEIGHTS_PATH))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    id_to_file, id_to_cat, ann_by_image = load_coco_annotations(VAL_COCO_PATH)

    for image_id in sample_ids:
        file_name = id_to_file.get(image_id)
        if not file_name:
            continue
        image_path = TRAIN_IMAGE_DIR / file_name
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")

        # Ground truth annotations
        gt_annotations = ann_by_image.get(image_id, [])

        # Predictions
        result = model.predict(
            source=str(image_path),
            device=device,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            verbose=False,
        )[0]

        pred_annotations: List[Annotation] = []
        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), cls_idx in zip(xyxy, cls):
                class_idx_int = int(cls_idx)
                category_id = inverse_mapping.get(class_idx_int, class_idx_int)
                category_name = id_to_cat.get(category_id, str(category_id))
                bbox = xyxy_to_xywh(x1, y1, x2, y2, img_w=image.width, img_h=image.height)
                pred_annotations.append(
                    Annotation(
                        category_id=category_id,
                        category_name=category_name,
                        bbox=bbox,
                    )
                )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        draw_annotations(axes[0], image, gt_annotations, f"GT: {image_id}", edge_color="#1f77b4")
        draw_annotations(axes[1], image, pred_annotations, "Pred", edge_color="#d62728")

        fig.tight_layout()
        output_path = OUTPUT_DIR / f"compare_{image_id}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"[INFO] 시각화 저장: {output_path}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Ground truth vs prediction visualization generator")
    parser.add_argument(
        "--count",
        type=int,
        default=SAMPLE_COUNT,
        help="랜덤으로 추출할 샘플 수 (기본 6)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="",
        help="콤마로 구분된 image_id 목록. 지정 시 랜덤 샘플 대신 해당 목록을 사용",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="랜덤 시드 (기본 2024)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_to_file, _, _ = load_coco_annotations(VAL_COCO_PATH)
    image_ids = sorted(id_to_file.keys())

    if args.images:
        sample_ids = []
        for token in args.images.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                image_id = int(token)
            except ValueError as exc:
                raise ValueError(f"image_id는 정수여야 합니다: '{token}'") from exc
            if image_id not in id_to_file:
                raise ValueError(f"val.json에 존재하지 않는 image_id 입니다: {image_id}")
            sample_ids.append(image_id)
        if not sample_ids:
            raise ValueError("유효한 image_id가 제공되지 않았습니다.")
    else:
        rng = random.Random(args.seed)
        rng.shuffle(image_ids)
        sample_ids = image_ids[: max(0, min(args.count, len(image_ids)))]

    inverse_mapping = load_inverse_mapping(CLASS_MAPPING_PATH)
    generate_visualizations(sample_ids, inverse_mapping)


if __name__ == "__main__":
    main()
