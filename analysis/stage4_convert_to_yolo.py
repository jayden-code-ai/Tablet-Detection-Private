#!/usr/bin/env python3
"""
Stage 4: COCO → YOLO 포맷 변환 및 학습용 데이터셋 구성 스크립트.

- 입력: Stage 3에서 생성한 COCO JSON (`stage3_dataset_artifacts/coco/train.json`, `val.json`)
- 출력: YOLO 형식 라벨(txt), 이미지 심볼릭 링크, 클래스 매핑, 데이터셋 YAML, 보고서 및 시각화

모든 경로는 절대 경로를 사용하며, 출력은
`/mnt/nas/jayden_code/Tablet-Detection-Private/stage4_yolo_dataset` 하위에 저장된다.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    font_manager = None  # type: ignore[assignment]
    rcParams = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False


PROJECT_ROOT = Path("/mnt/nas/jayden_code/Tablet-Detection-Private")
STAGE3_COCO_DIR = PROJECT_ROOT / "stage3_dataset_artifacts" / "coco"

TRAIN_IMAGES_DIR = Path("/mnt/nas/jayden_code/ai05-level1-project/train_images")

OUTPUT_ROOT = PROJECT_ROOT / "stage4_yolo_dataset"
IMAGES_DIR = OUTPUT_ROOT / "images"
LABELS_DIR = OUTPUT_ROOT / "labels"
REPORT_DIR = OUTPUT_ROOT / "reports"
FIGURE_DIR = OUTPUT_ROOT / "figures"

SPLITS = {
    "train": STAGE3_COCO_DIR / "train.json",
    "val": STAGE3_COCO_DIR / "val.json",
}


def ensure_directories() -> None:
    """출력용 디렉터리를 생성한다."""
    for path in [IMAGES_DIR, LABELS_DIR, REPORT_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)


def configure_korean_font() -> None:
    """matplotlib 한글 폰트를 설정한다."""
    if not MATPLOTLIB_AVAILABLE:
        return

    preferred_keywords = [
        "nanumsquareround",
        "nanumsquare",
        "nanumbarungothic",
        "nanumgothic",
        "notosanscjkkr",
        "notosanscjk",
        "notoserifcjkkr",
        "undotum",
        "unbatang",
    ]

    system_fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font_path in system_fonts:
        lower = font_path.lower().replace(" ", "")
        if any(keyword in lower for keyword in preferred_keywords):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                continue

    available = {font.name for font in font_manager.fontManager.ttflist}
    for keyword in preferred_keywords:
        for name in available:
            if keyword in name.lower().replace(" ", ""):
                rcParams["font.family"] = name
                rcParams["axes.unicode_minus"] = False
                print(f"[INFO] Stage4 matplotlib 폰트: {name}")
                return

    rcParams["axes.unicode_minus"] = False
    print("[WARN] Stage4: 한글 폰트를 찾지 못했습니다. 기본 폰트 사용.")


def load_coco(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_class_mapping(coco_dict: Dict[str, object]) -> Dict[int, int]:
    """COCO 카테고리 ID를 0부터 시작하는 연속 인덱스로 매핑한다."""
    cat_ids = sorted(category["id"] for category in coco_dict["categories"])
    return {cat_id: idx for idx, cat_id in enumerate(cat_ids)}


def create_symlink(target: Path, link_path: Path) -> None:
    """이미지 파일을 가리키는 심볼릭 링크를 생성한다."""
    if link_path.exists():
        if link_path.is_symlink():
            existing_target = Path(os.readlink(link_path))
            if existing_target == target:
                return
            link_path.unlink()
        else:
            raise FileExistsError(f"{link_path} 이미 존재하며 심볼릭 링크가 아닙니다.")
    link_path.symlink_to(target)


def coco_to_yolo_records(
    annotations: Iterable[Dict[str, object]],
    image_width: int,
    image_height: int,
) -> List[Tuple[int, float, float, float, float]]:
    """COCO bbox를 YOLO 형식으로 변환한다."""
    records = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0 or image_width <= 0 or image_height <= 0:
            continue

        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        w_ratio = w / image_width
        h_ratio = h / image_height

        records.append(
            (
                int(ann["category_id"]),
                x_center,
                y_center,
                w_ratio,
                h_ratio,
            )
        )
    return records


def write_label_file(path: Path, rows: List[Tuple[int, float, float, float, float]], mapping: Dict[int, int]) -> None:
    """YOLO 라벨 파일을 저장한다."""
    with path.open("w", encoding="utf-8") as handle:
        for category_id, x_center, y_center, w_ratio, h_ratio in rows:
            class_idx = mapping[category_id]
            handle.write(
                f"{class_idx} {x_center:.6f} {y_center:.6f} {w_ratio:.6f} {h_ratio:.6f}\n"
            )


def process_split(
    split_name: str,
    coco_dict: Dict[str, object],
    mapping: Dict[int, int],
) -> Dict[str, int]:
    """단일 분할(train/val)을 처리하고 통계를 반환한다."""
    images = {img["id"]: img for img in coco_dict["images"]}
    annotations_per_image: Dict[int, List[Dict[str, object]]] = {}
    for ann in coco_dict["annotations"]:
        annotations_per_image.setdefault(ann["image_id"], []).append(ann)

    images_processed = 0
    labels_written = 0
    annotation_counter = 0
    class_counter: Counter = Counter()

    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        target_image = TRAIN_IMAGES_DIR / file_name
        if not target_image.exists():
            raise FileNotFoundError(f"{target_image} 이미지가 존재하지 않습니다.")

        # 이미지 심볼릭 링크 생성
        link_path = IMAGES_DIR / split_name / file_name
        link_path.parent.mkdir(parents=True, exist_ok=True)
        create_symlink(target_image, link_path)

        # 라벨 생성
        label_path = LABELS_DIR / split_name / (Path(file_name).stem + ".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)

        rows = coco_to_yolo_records(
            annotations_per_image.get(image_id, []),
            image_width=width,
            image_height=height,
        )
        if not rows:
            continue

        write_label_file(label_path, rows, mapping)

        labels_written += 1
        images_processed += 1
        for category_id, *_ in rows:
            class_counter[mapping[category_id]] += 1
            annotation_counter += 1

    return {
        "images_processed": images_processed,
        "labels_written": labels_written,
        "annotations_written": annotation_counter,
        "unique_classes": len(class_counter),
    }


def build_dataset_yaml(categories: Dict[int, Dict[str, object]], mapping: Dict[int, int]) -> Dict[str, object]:
    """YOLO 데이터셋 YAML 내용을 구성한다."""
    names = [""] * len(mapping)
    for category_id, class_idx in mapping.items():
        names[class_idx] = categories[category_id]["name"]

    return {
        "path": str(OUTPUT_ROOT),
        "train": str(IMAGES_DIR / "train"),
        "val": str(IMAGES_DIR / "val"),
        "nc": len(names),
        "names": names,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def build_summary_text(
    overall_mapping: Dict[int, int],
    split_stats: Dict[str, Dict[str, int]],
) -> str:
    lines = [
        "=== Stage 4 YOLO 변환 요약 ===",
        f"총 클래스 수: {len(overall_mapping)}",
    ]
    for split, stats in split_stats.items():
        lines.append("")
        lines.append(f"[{split}]")
        lines.append(f"  이미지 수: {stats['images_processed']}")
        lines.append(f"  라벨 수: {stats['labels_written']}")
        lines.append(f"  어노테이션 수: {stats['annotations_written']}")
        lines.append(f"  등장 클래스 수: {stats['unique_classes']}")
    return "\n".join(lines)


def generate_class_distribution_figure(
    class_counters: Dict[str, Counter],
    class_names: List[str],
) -> List[str]:
    if not MATPLOTLIB_AVAILABLE:
        return []

    configure_korean_font()

    figure_paths: List[str] = []

    for split, counter in class_counters.items():
        if not counter:
            continue
        labels = []
        values = []
        for class_idx, count in counter.most_common(20):
            labels.append(class_names[class_idx])
            values.append(count)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels[::-1], values[::-1], color="#1f77b4" if split == "train" else "#ff7f0e")
        ax.set_title(f"{split.upper()} 상위 20 클래스 어노테이션 수")
        ax.set_xlabel("어노테이션 수")
        fig.tight_layout()
        path = FIGURE_DIR / f"{split}_class_distribution.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_paths.append(str(path))

    return figure_paths


def main() -> None:
    ensure_directories()

    train_coco = load_coco(SPLITS["train"])
    val_coco = load_coco(SPLITS["val"])

    # 전체 카테고리 정보는 train/val 동일하다고 가정
    categories = {
        category["id"]: {"name": category["name"], "supercategory": category.get("supercategory")}
        for category in train_coco["categories"]
    }
    class_mapping = build_class_mapping(train_coco)

    split_stats: Dict[str, Dict[str, int]] = {}
    class_counters: Dict[str, Counter] = {}

    for split_name, coco_dict in [("train", train_coco), ("val", val_coco)]:
        stats = process_split(split_name, coco_dict, class_mapping)
        split_stats[split_name] = stats

        counter: Counter = Counter()
        for ann in coco_dict["annotations"]:
            class_idx = class_mapping[ann["category_id"]]
            counter[class_idx] += 1
        class_counters[split_name] = counter

    # dataset.yaml 작성
    dataset_yaml = build_dataset_yaml(categories, class_mapping)
    yaml_lines = [
        f"path: {OUTPUT_ROOT}",
        f"train: {dataset_yaml['train']}",
        f"val: {dataset_yaml['val']}",
        f"nc: {dataset_yaml['nc']}",
        "names:",
    ]
    for idx, name in enumerate(dataset_yaml["names"]):
        yaml_lines.append(f"  {idx}: {name}")
    save_text(OUTPUT_ROOT / "dataset.yaml", "\n".join(yaml_lines) + "\n")

    # 클래스 매핑 저장
    mapping_payload = {
        "class_mapping": {str(cat_id): idx for cat_id, idx in class_mapping.items()},
        "class_names": dataset_yaml["names"],
    }
    save_json(REPORT_DIR / "class_mapping.json", mapping_payload)

    # 요약 저장
    summary_text = build_summary_text(class_mapping, split_stats)
    save_text(REPORT_DIR / "conversion_summary.txt", summary_text)
    save_json(REPORT_DIR / "conversion_summary.json", {"split_stats": split_stats})

    figures = generate_class_distribution_figure(class_counters, dataset_yaml["names"])

    print(summary_text)
    if figures:
        print("")
        print("생성된 시각화 파일:")
        for path in figures:
            print(f"- {path}")
    elif not MATPLOTLIB_AVAILABLE:
        print("")
        print("[WARN] matplotlib 미설치로 시각화를 생성하지 못했습니다.")


if __name__ == "__main__":
    main()
