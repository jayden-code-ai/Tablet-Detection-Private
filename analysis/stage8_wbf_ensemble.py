#!/usr/bin/env python3
"""
Stage 8: Weighted Boxes Fusion 앙상블 스크립트.

입력: 여러 YOLO 제출 CSV
출력: WBF를 적용한 최종 제출 CSV
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ensemble_boxes import weighted_boxes_fusion
from PIL import Image

PROJECT_ROOT = Path("/mnt/nas/jayden_code/Tablet-Detection-Private")
TEST_IMAGE_DIR = Path("/mnt/nas/jayden_code/ai05-level1-project/test_images")
OUTPUT_DIR = PROJECT_ROOT / "stage6_submission_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: int


def load_submission(path: Path) -> Dict[int, List[Detection]]:
    """제출 CSV를 읽어 이미지별 Detection 목록으로 변환."""
    per_image: Dict[int, List[Detection]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        required = {"image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}에 필요한 컬럼이 없습니다. (필요: {required})")
        for row in reader:
            image_id = int(row["image_id"])
            category_id = int(row["category_id"])
            x = float(row["bbox_x"])
            y = float(row["bbox_y"])
            w = float(row["bbox_w"])
            h = float(row["bbox_h"])
            score = float(row["score"])
            det = Detection(x1=x, y1=y, x2=x + w, y2=y + h, score=score, label=category_id)
            per_image[image_id].append(det)
    return per_image


def gather_image_sizes(image_dir: Path) -> Dict[int, Tuple[int, int]]:
    """이미지 파일에서 (width, height)를 읽는다."""
    sizes: Dict[int, Tuple[int, int]] = {}
    for path in image_dir.glob("*.png"):
        stem = path.stem
        try:
            image_id = int(stem)
        except ValueError:
            continue
        with Image.open(path) as img:
            sizes[image_id] = img.size  # (width, height)
    return sizes


def normalize(det: Detection, width: int, height: int) -> Tuple[float, float, float, float]:
    return (
        det.x1 / width,
        det.y1 / height,
        det.x2 / width,
        det.y2 / height,
    )


def denormalize(
    box: Sequence[float], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1 = max(0.0, min(box[0], 1.0)) * width
    y1 = max(0.0, min(box[1], 1.0)) * height
    x2 = max(0.0, min(box[2], 1.0)) * width
    y2 = max(0.0, min(box[3], 1.0)) * height
    x = max(0.0, min(x1, width - 1))
    y = max(0.0, min(y1, height - 1))
    w = max(1.0, min(x2 - x1, width - x))
    h = max(1.0, min(y2 - y1, height - y))
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def run_wbf(
    submissions: List[Dict[int, List[Detection]]],
    weights: List[float],
    image_sizes: Dict[int, Tuple[int, int]],
    iou_thr: float,
    skip_box_thr: float,
) -> Dict[int, List[Detection]]:
    """WBF를 적용해 이미지별 Detection을 반환한다."""
    ensemble: Dict[int, List[Detection]] = {}
    image_ids = set().union(*[sub.keys() for sub in submissions])

    for image_id in sorted(image_ids):
        width, height = image_sizes.get(image_id, (None, None))
        if width is None or height is None:
            continue

        boxes_list = []
        scores_list = []
        labels_list = []

        for sub in submissions:
            dets = sub.get(image_id, [])
            if not dets:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue
            boxes_list.append([normalize(det, width, height) for det in dets])
            scores_list.append([det.score for det in dets])
            labels_list.append([det.label for det in dets])

        if not any(scores_list):
            ensemble[image_id] = []
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        image_dets: List[Detection] = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x, y, w, h = denormalize(box, width, height)
            image_dets.append(
                Detection(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    score=float(round(float(score), 6)),
                    label=int(label),
                )
            )

        image_dets.sort(key=lambda d: d.score, reverse=True)
        ensemble[image_id] = image_dets

    return ensemble


def write_submission_csv(
    detections: Dict[int, List[Detection]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as csvfile:
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
        annotation_id = 1
        for image_id in sorted(detections.keys()):
            for det in detections[image_id]:
                bbox_x = det.x1
                bbox_y = det.y1
                bbox_w = det.x2 - det.x1
                bbox_h = det.y2 - det.y1
                writer.writerow(
                    [
                        annotation_id,
                        image_id,
                        det.label,
                        int(bbox_x),
                        int(bbox_y),
                        int(bbox_w),
                        int(bbox_h),
                        f"{det.score:.6f}",
                    ]
                )
                annotation_id += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted Boxes Fusion ensemble for Tablet Detection")
    parser.add_argument(
        "--submissions",
        type=Path,
        nargs="+",
        required=True,
        help="앙상블에 사용할 제출 CSV 경로들",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        help="각 제출에 대한 가중치 (기본: 동일 가중치)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "submission_ensemble_wbf.csv",
        help="결과 저장 경로",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.55,
        help="WBF IoU threshold",
    )
    parser.add_argument(
        "--skip-box-thr",
        type=float,
        default=0.001,
        help="WBF skip box threshold",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_IMAGE_DIR,
        help="테스트 이미지 디렉터리",
    )
    args = parser.parse_args()

    if args.weights and len(args.weights) != len(args.submissions):
        raise ValueError("weights의 길이는 submissions와 동일해야 합니다.")

    submissions = [load_submission(path) for path in args.submissions]
    weights = args.weights if args.weights else [1.0] * len(submissions)

    image_sizes = gather_image_sizes(args.test_dir)
    detections = run_wbf(
        submissions,
        weights=weights,
        image_sizes=image_sizes,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission_csv(detections, args.output)

    total = sum(len(dets) for dets in detections.values())
    print(f"[INFO] 앙상블 제출을 {args.output}에 저장했습니다. 총 박스 수: {total}")


if __name__ == "__main__":
    main()
