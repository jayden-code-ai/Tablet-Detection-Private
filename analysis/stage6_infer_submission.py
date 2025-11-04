#!/usr/bin/env python3
"""
Stage 6: YOLOv8 추론을 통해 Kaggle 제출 파일 생성.

- 입력: Stage 1에서 학습한 best.pt 체크포인트
- 출력: `stage6_submission_artifacts/submission_epoch44.csv`
        (annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path("/mnt/nas/jayden_code/Tablet-Detection-Private")
DEFAULT_TEST_IMAGE_DIR = Path("/mnt/nas/jayden_code/ai05-level1-project/test_images")
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "stage5_yolov8l_runs" / "yolov8l_stage12" / "weights" / "best.pt"
CLASS_MAPPING_PATH = PROJECT_ROOT / "stage4_yolo_dataset" / "reports" / "class_mapping.json"

OUTPUT_DIR = PROJECT_ROOT / "stage6_submission_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SUBMISSION_PATH = OUTPUT_DIR / "submission_epoch44.csv"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "submission_summary.json"


@dataclass
class Prediction:
    annotation_id: int
    image_id: int
    category_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    score: float


def load_inverse_mapping(mapping_path: Path) -> Dict[int, int]:
    """YOLO class index -> original category_id 매핑을 구성한다."""
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_mapping = payload["class_mapping"]  # original_id(str) -> idx
    inverse: Dict[int, int] = {}
    for original_id_str, class_idx in class_mapping.items():
        inverse[int(class_idx)] = int(original_id_str)
    return inverse


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def xyxy_to_xywh(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """
    좌표를 int(x, y, w, h)로 변환하고 이미지 경계를 벗어나지 않도록 클램핑한다.
    """
    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, x1, img_w - 1)
    y2 = clamp(y2, y1, img_h - 1)

    w = clamp(x2 - x1, 1, img_w - x1)
    h = clamp(y2 - y1, 1, img_h - y1)

    return int(round(x1)), int(round(y1)), int(round(w)), int(round(h))


def iter_chunks(sequence: List[Path], size: int) -> Iterable[List[Path]]:
    for idx in range(0, len(sequence), size):
        yield sequence[idx : idx + size]


def collect_predictions(
    model: YOLO,
    class_map: Dict[int, int],
    image_paths: Iterable[Path],
    *,
    conf_thresh: float,
    iou_thresh: float,
    chunk_size: int,
    batch_size: int,
) -> List[Prediction]:
    """
    YOLO 추론 결과를 Kaggle 제출용 포맷으로 수집한다.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictions: List[Prediction] = []
    annotation_counter = 1

    image_paths_list = list(image_paths)
    for chunk in iter_chunks(image_paths_list, size=chunk_size):
        results = model.predict(
            source=[str(p) for p in chunk],
            device=device,
            imgsz=640,
            conf=conf_thresh,
            iou=iou_thresh,
            stream=True,
            save=False,
            verbose=False,
            batch=batch_size,
        )

        for result in results:
            image_path = Path(result.path)
            stem = image_path.stem
            try:
                image_id = int(stem)
            except ValueError as exc:
                raise ValueError(f"테스트 이미지 파일명이 정수여야 합니다: {image_path}") from exc

            h, w = result.orig_shape

            boxes = result.boxes
            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf_scores = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls_idx, score in zip(xyxy, cls, conf_scores):
                class_idx_int = int(cls_idx)
                original_category_id = class_map.get(class_idx_int, class_idx_int)

                bbox_x, bbox_y, bbox_w, bbox_h = xyxy_to_xywh(x1, y1, x2, y2, img_w=w, img_h=h)

                predictions.append(
                    Prediction(
                        annotation_id=annotation_counter,
                        image_id=image_id,
                        category_id=original_category_id,
                        bbox_x=bbox_x,
                        bbox_y=bbox_y,
                        bbox_w=bbox_w,
                        bbox_h=bbox_h,
                        score=float(round(float(score), 6)),
                    )
                )
                annotation_counter += 1

    predictions.sort(key=lambda p: (p.image_id, -p.score))
    for new_id, pred in enumerate(predictions, start=1):
        pred.annotation_id = new_id
    return predictions


def write_submission(predictions: List[Prediction], path: Path) -> None:
    """CSV 파일로 제출 결과를 기록한다."""
    with path.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "annotation_id",
                "image_id",
                "category_id",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "score",
            ]
        )
        for pred in predictions:
            writer.writerow(
                [
                    pred.annotation_id,
                    pred.image_id,
                    pred.category_id,
                    pred.bbox_x,
                    pred.bbox_y,
                    pred.bbox_w,
                    pred.bbox_h,
                    f"{pred.score:.6f}",
                ]
            )


def write_summary(predictions: List[Prediction], path: Path) -> None:
    """간단한 요약 정보를 JSON으로 저장한다."""
    by_image: Dict[int, int] = {}
    for pred in predictions:
        by_image[pred.image_id] = by_image.get(pred.image_id, 0) + 1

    summary = {
        "total_predictions": len(predictions),
        "images_with_predictions": len(by_image),
        "mean_predictions_per_image": (
            sum(by_image.values()) / len(by_image) if by_image else 0.0
        ),
        "max_predictions_per_image": max(by_image.values()) if by_image else 0,
        "min_predictions_per_image": min(by_image.values()) if by_image else 0,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 6 YOLO inference to Kaggle submission")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="YOLO 가중치 경로 (기본: Stage1 yolov8l best.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SUBMISSION_PATH,
        help="저장할 제출 CSV 경로",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="요약 정보를 저장할 JSON 경로",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_IMAGE_DIR,
        help="테스트 이미지 디렉터리 절대 경로",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--chunk-size", type=int, default=32, help="추론 시 한 번에 처리할 이미지 개수")
    parser.add_argument("--batch-size", type=int, default=1, help="YOLO 모델 배치 크기")
    args = parser.parse_args()

    weights_path: Path = args.weights
    output_path: Path = args.output
    summary_path: Path = args.summary
    test_image_dir: Path = args.test_dir

    if not weights_path.exists():
        raise FileNotFoundError(f"모델 가중치를 찾을 수 없습니다: {weights_path}")
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(f"클래스 매핑 파일을 찾을 수 없습니다: {CLASS_MAPPING_PATH}")
    if not test_image_dir.exists():
        raise FileNotFoundError(f"테스트 이미지 디렉터리를 찾을 수 없습니다: {test_image_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    inverse_mapping = load_inverse_mapping(CLASS_MAPPING_PATH)

    test_images = sorted(test_image_dir.glob("*.png"))
    if not test_images:
        raise FileNotFoundError(f"테스트 이미지가 존재하지 않습니다: {test_image_dir}")

    predictions = collect_predictions(
        model,
        inverse_mapping,
        test_images,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        chunk_size=max(1, args.chunk_size),
        batch_size=max(1, args.batch_size),
    )
    write_submission(predictions, output_path)
    write_summary(predictions, summary_path)

    print(f"[INFO] 총 {len(predictions)}개의 예측을 {output_path}에 저장했습니다.")
    print(f"[INFO] 요약 정보는 {summary_path}에서 확인할 수 있습니다.")


if __name__ == "__main__":
    main()
