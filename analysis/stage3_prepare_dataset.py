#!/usr/bin/env python3
"""
Stage 3: 중복 제거 기반 COCO 통합 및 학습/검증 분할 스크립트.

주요 기능
---------
1. `/mnt/nas/jayden_code/ai05-level1-project/train_annotations` 내 개별 약품 폴더의
   COCO JSON을 읽어 이미지별 어노테이션을 통합한다.
   - `train_annotations_merged.json`은 중복을 유발하므로 제외한다.
   - 동일 이미지 파일에 대해 여러 JSON에서 수집된 어노테이션을 합치며,
     `(카테고리, 바운딩 박스)` 조합의 중복은 제거한다.
2. 고유 이미지/어노테이션/카테고리 정보를 새 COCO 형식으로 재작성한다.
   - 새 `image_id`/`annotation_id`는 1부터 재부여한다.
3. 이미지 단위 8:2(학습:검증) 분할을 수행하고 각각 COCO JSON을 생성한다.
   - 분할은 `random.Random(2024)`로 시드 고정해 재현성을 보장한다.
4. 요약 리포트와 시각화(한글 폰트 적용)를
   `/mnt/nas/jayden_code/Tablet-Detection-Private/stage3_dataset_artifacts` 하위에 저장한다.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - matplotlib 외부 의존성
    plt = None  # type: ignore[assignment]
    font_manager = None  # type: ignore[assignment]
    rcParams = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False


DATA_ROOT = Path("/mnt/nas/jayden_code/ai05-level1-project")
TRAIN_IMAGE_DIR = DATA_ROOT / "train_images"
ANNOTATION_ROOT = DATA_ROOT / "train_annotations"
MERGED_FILENAME = "train_annotations_merged.json"

OUTPUT_ROOT = Path("/mnt/nas/jayden_code/Tablet-Detection-Private/stage3_dataset_artifacts")
COCO_DIR = OUTPUT_ROOT / "coco"
REPORT_DIR = OUTPUT_ROOT / "reports"
FIGURE_DIR = OUTPUT_ROOT / "figures"

TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"
SPLIT_RATIO = 0.8
RANDOM_SEED = 2024


@dataclass
class AggregatedAnnotation:
    category_id: int
    bbox: Tuple[float, float, float, float]
    source_files: List[str] = field(default_factory=list)


@dataclass
class AggregatedImage:
    width: int
    height: int
    sources: List[str] = field(default_factory=list)
    annotations: Dict[Tuple[int, Tuple[float, float, float, float]], AggregatedAnnotation] = field(
        default_factory=dict
    )


def ensure_directories() -> None:
    """출력 경로를 생성한다."""
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def configure_korean_font() -> Optional[str]:
    """matplotlib 시각화를 위해 한글 지원 폰트를 설정한다."""
    if not MATPLOTLIB_AVAILABLE:
        return None

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
    registered_paths: List[str] = []
    for font_path in system_fonts:
        lower_path = font_path.lower()
        if any(keyword in lower_path for keyword in preferred_keywords):
            if font_path not in registered_paths:
                try:
                    font_manager.fontManager.addfont(font_path)
                    registered_paths.append(font_path)
                except Exception:
                    continue

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for keyword in preferred_keywords:
        for font_name in available_fonts:
            if keyword in font_name.lower().replace(" ", ""):
                rcParams["font.family"] = font_name
                rcParams["axes.unicode_minus"] = False
                return font_name

    rcParams["axes.unicode_minus"] = False
    return None


def gather_annotation_paths(root: Path) -> List[Path]:
    """중복을 유발하는 통합 파일을 제외한 JSON 경로 목록을 생성한다."""
    paths = []
    for json_path in sorted(root.rglob("*.json")):
        if json_path.name == MERGED_FILENAME:
            continue
        paths.append(json_path)
    return paths


def aggregate_annotations(annotation_paths: Iterable[Path]) -> Tuple[
    Dict[str, AggregatedImage],
    Dict[int, Dict[str, object]],
    Dict[str, List[str]],
    List[Dict[str, object]],
]:
    """
    이미지 파일명 기준으로 어노테이션을 통합한다.

    반환값:
      - images_by_file: 파일명 -> AggregatedImage
      - categories: category_id -> {name, supercategory, sources}
      - image_conflicts: 파일명 -> [충돌 메시지]
      - anomaly_records: 예외 상황 로그용 목록
    """
    images_by_file: Dict[str, AggregatedImage] = {}
    categories: Dict[int, Dict[str, object]] = {}
    image_conflicts: Dict[str, List[str]] = defaultdict(list)
    anomaly_records: List[Dict[str, object]] = []

    for json_path in annotation_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            anomaly_records.append(
                {
                    "issue": "json_decode_error",
                    "path": str(json_path),
                    "error": str(exc),
                }
            )
            continue

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        cats = data.get("categories", [])

        local_image_map = {img.get("id"): img for img in images if "id" in img}

        for cat in cats:
            cat_id = cat.get("id")
            name = cat.get("name")
            if cat_id is None or name is None:
                anomaly_records.append(
                    {
                        "issue": "category_missing_fields",
                        "path": str(json_path),
                        "category": cat,
                    }
                )
                continue

            entry = categories.setdefault(
                int(cat_id),
                {
                    "name": name,
                    "supercategory": cat.get("supercategory"),
                    "sources": [],
                },
            )
            entry["sources"].append(str(json_path))
            if entry["name"] != name:
                image_conflicts.setdefault("category_name_conflict", []).append(
                    f"category_id {cat_id} name mismatch: '{entry['name']}' vs '{name}' @ {json_path}"
                )

        for ann in annotations:
            ann_id = ann.get("id")
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")
            bbox = ann.get("bbox")
            if (
                ann_id is None
                or image_id is None
                or category_id is None
                or not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
            ):
                anomaly_records.append(
                    {
                        "issue": "annotation_missing_fields",
                        "path": str(json_path),
                        "annotation": ann,
                    }
                )
                continue

            image_info = local_image_map.get(image_id)
            if image_info is None:
                anomaly_records.append(
                    {
                        "issue": "image_missing_for_annotation",
                        "path": str(json_path),
                        "annotation_id": ann_id,
                        "image_id": image_id,
                    }
                )
                continue

            file_name = image_info.get("file_name")
            width = image_info.get("width")
            height = image_info.get("height")
            if file_name is None or width is None or height is None:
                anomaly_records.append(
                    {
                        "issue": "image_missing_fields",
                        "path": str(json_path),
                        "image": image_info,
                    }
                )
                continue

            bbox_tuple = tuple(float(value) for value in bbox)
            image_entry = images_by_file.setdefault(
                file_name,
                AggregatedImage(
                    width=int(width),
                    height=int(height),
                    sources=[str(json_path)],
                ),
            )

            if image_entry.width != int(width) or image_entry.height != int(height):
                image_conflicts.setdefault(file_name, []).append(
                    f"Dimension mismatch ({image_entry.width}x{image_entry.height} vs {width}x{height})"
                )

            if str(json_path) not in image_entry.sources:
                image_entry.sources.append(str(json_path))

            key = (int(category_id), bbox_tuple)
            record = image_entry.annotations.get(key)
            if record is None:
                image_entry.annotations[key] = AggregatedAnnotation(
                    category_id=int(category_id),
                    bbox=bbox_tuple,
                    source_files=[str(json_path)],
                )
            else:
                record.source_files.append(str(json_path))

    return images_by_file, categories, image_conflicts, anomaly_records


def reindex_coco(
    images_by_file: Dict[str, AggregatedImage],
    categories: Dict[int, Dict[str, object]],
) -> Tuple[Dict[str, object], Dict[str, int], Dict[int, Dict[str, object]]]:
    """이미지/어노테이션을 재색인한 COCO 딕셔너리를 반환한다."""
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []

    file_to_image_id: Dict[str, int] = {}
    annotation_id = 1
    image_id = 1

    for file_name in sorted(images_by_file.keys()):
        agg_image = images_by_file[file_name]
        file_to_image_id[file_name] = image_id

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": agg_image.width,
                "height": agg_image.height,
            }
        )

        for annotation in agg_image.annotations.values():
            x, y, w, h = annotation.bbox
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation.category_id,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

        image_id += 1

    categories_list = [
        {"id": cat_id, "name": info["name"], "supercategory": info.get("supercategory")}
        for cat_id, info in sorted(categories.items())
    ]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
        "type": "instances",
    }
    return coco_dict, file_to_image_id, categories


def split_dataset(file_names: List[str]) -> Dict[str, List[str]]:
    """이미지 파일명을 기반으로 8:2 분할을 수행한다."""
    rng = random.Random(RANDOM_SEED)
    shuffled = list(file_names)
    rng.shuffle(shuffled)

    split_index = int(len(shuffled) * SPLIT_RATIO)
    return {
        TRAIN_SPLIT_NAME: sorted(shuffled[:split_index]),
        VAL_SPLIT_NAME: sorted(shuffled[split_index:]),
    }


def filter_coco_by_files(
    coco_dict: Dict[str, object], target_files: Iterable[str], file_to_image_id: Dict[str, int]
) -> Dict[str, object]:
    """COCO 딕셔너리를 이미지 파일명 목록으로 필터링한다."""
    target_files_set = set(target_files)
    target_image_ids = {file_to_image_id[file_name] for file_name in target_files_set}

    images = [img for img in coco_dict["images"] if img["id"] in target_image_ids]
    annotations = [ann for ann in coco_dict["annotations"] if ann["image_id"] in target_image_ids]

    return {
        "images": images,
        "annotations": annotations,
        "categories": coco_dict["categories"],
        "type": coco_dict.get("type", "instances"),
    }


def image_annotation_statistics(images_by_file: Dict[str, AggregatedImage]) -> Dict[str, object]:
    """이미지별 어노테이션 통계를 계산한다."""
    counts = [len(img.annotations) for img in images_by_file.values()]
    sources = [len(img.sources) for img in images_by_file.values()]

    stats = {
        "min_annotations_per_image": int(np.min(counts)),
        "max_annotations_per_image": int(np.max(counts)),
        "mean_annotations_per_image": float(np.mean(counts)),
        "median_annotations_per_image": float(np.median(counts)),
        "images_with_multiple_annotations": int(np.sum(np.array(counts) > 1)),
        "min_sources_per_image": int(np.min(sources)),
        "max_sources_per_image": int(np.max(sources)),
        "mean_sources_per_image": float(np.mean(sources)),
    }
    return stats


def category_distribution(
    coco_dict: Dict[str, object]
) -> Counter:
    """카테고리별 어노테이션 수를 카운트한다."""
    counter: Counter = Counter()
    for ann in coco_dict["annotations"]:
        counter[ann["category_id"]] += 1
    return counter


def generate_figures(
    overall_counts: Counter,
    train_counts: Counter,
    val_counts: Counter,
    categories: Dict[int, Dict[str, object]],
) -> List[str]:
    """카테고리 분포와 이미지당 어노테이션 수를 시각화한다."""
    if not MATPLOTLIB_AVAILABLE:
        return []

    configured_font = configure_korean_font()
    if configured_font:
        print(f"[INFO] Stage3 matplotlib 폰트: {configured_font}")
    else:
        print("[WARN] Stage3: 한글 폰트를 찾지 못했습니다. 기본 폰트 사용.")

    figure_paths: List[str] = []

    def top_items(counter: Counter, top_n: int = 20) -> Tuple[List[str], List[int]]:
        items = counter.most_common(top_n)
        labels = []
        values = []
        for cat_id, count in items:
            label = categories.get(cat_id, {}).get("name") or str(cat_id)
            labels.append(label)
            values.append(count)
        return labels, values

    labels_all, values_all = top_items(overall_counts, top_n=20)
    labels_train, values_train = top_items(train_counts, top_n=20)
    labels_val, values_val = top_items(val_counts, top_n=20)

    # 전체 분포
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels_all[::-1], values_all[::-1], color="#2ca02c")
    ax.set_title("전체 카테고리 어노테이션 수 상위 20")
    ax.set_xlabel("어노테이션 수")
    fig.tight_layout()
    path_all = FIGURE_DIR / "overall_category_distribution.png"
    fig.savefig(path_all, dpi=200)
    plt.close(fig)
    figure_paths.append(str(path_all))

    # Train/Val 비교
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=False)
    axes[0].barh(labels_train[::-1], values_train[::-1], color="#1f77b4")
    axes[0].set_title("Train 상위 20 카테고리")
    axes[0].set_xlabel("어노테이션 수")
    axes[1].barh(labels_val[::-1], values_val[::-1], color="#ff7f0e")
    axes[1].set_title("Val 상위 20 카테고리")
    axes[1].set_xlabel("어노테이션 수")
    fig.tight_layout()
    path_split = FIGURE_DIR / "split_category_distribution.png"
    fig.savefig(path_split, dpi=200)
    plt.close(fig)
    figure_paths.append(str(path_split))

    return figure_paths


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text_summary(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_text_summary(
    coco_dict: Dict[str, object],
    train_coco: Dict[str, object],
    val_coco: Dict[str, object],
    stats: Dict[str, object],
    split_map: Dict[str, List[str]],
    conflicts: Dict[str, List[str]],
    anomalies: List[Dict[str, object]],
) -> str:
    lines = [
        "=== Stage 3 Dataset Preparation Summary ===",
        f"총 이미지 수: {len(coco_dict['images'])}",
        f"총 어노테이션 수: {len(coco_dict['annotations'])}",
        f"카테고리 수: {len(coco_dict['categories'])}",
        "",
        f"Train 이미지 수: {len(train_coco['images'])}",
        f"Train 어노테이션 수: {len(train_coco['annotations'])}",
        f"Val 이미지 수: {len(val_coco['images'])}",
        f"Val 어노테이션 수: {len(val_coco['annotations'])}",
        "",
        "이미지당 어노테이션 통계:",
        f"  최소: {stats['min_annotations_per_image']}",
        f"  최대: {stats['max_annotations_per_image']}",
        f"  평균: {stats['mean_annotations_per_image']:.3f}",
        f"  중앙값: {stats['median_annotations_per_image']:.3f}",
        f"  2개 이상 어노테이션 보유 이미지 수: {stats['images_with_multiple_annotations']}",
        "",
        f"이미지별 참조 JSON 개수 최소/최대/평균: "
        f"{stats['min_sources_per_image']} / {stats['max_sources_per_image']} / {stats['mean_sources_per_image']:.3f}",
    ]

    if conflicts:
        lines.append("")
        lines.append(f"경고: {sum(len(v) for v in conflicts.values())}건의 충돌 기록 존재")
        for key, messages in conflicts.items():
            lines.append(f"- {key}: {len(messages)}건")

    if anomalies:
        lines.append("")
        lines.append(f"주의: {len(anomalies)}건의 파싱 예외가 발생했습니다. 상세 정보는 JSON 리포트 확인.")

    lines.append("")
    lines.append("분할 예시 (상위 5개 파일명):")
    for split_name in [TRAIN_SPLIT_NAME, VAL_SPLIT_NAME]:
        sample = split_map[split_name][:5]
        lines.append(f"- {split_name}: {', '.join(sample)}")

    return "\n".join(lines)


def main() -> None:
    ensure_directories()

    annotation_paths = gather_annotation_paths(ANNOTATION_ROOT)
    (
        images_by_file,
        categories,
        conflicts,
        anomalies,
    ) = aggregate_annotations(annotation_paths)

    coco_dict, file_to_image_id, categories_registry = reindex_coco(images_by_file, categories)
    split_map = split_dataset(list(images_by_file.keys()))

    train_coco = filter_coco_by_files(coco_dict, split_map[TRAIN_SPLIT_NAME], file_to_image_id)
    val_coco = filter_coco_by_files(coco_dict, split_map[VAL_SPLIT_NAME], file_to_image_id)

    stats = image_annotation_statistics(images_by_file)

    overall_counts = category_distribution(coco_dict)
    train_counts = category_distribution(train_coco)
    val_counts = category_distribution(val_coco)

    # 저장
    save_json(COCO_DIR / "deduplicated_all.json", coco_dict)
    save_json(COCO_DIR / "train.json", train_coco)
    save_json(COCO_DIR / "val.json", val_coco)
    save_json(REPORT_DIR / "category_registry.json", categories_registry)
    save_json(REPORT_DIR / "split_map.json", split_map)
    save_json(REPORT_DIR / "conflicts.json", conflicts)
    save_json(REPORT_DIR / "anomalies.json", {"records": anomalies})

    summary_payload = {
        "total_images": len(coco_dict["images"]),
        "total_annotations": len(coco_dict["annotations"]),
        "total_categories": len(coco_dict["categories"]),
        "train_images": len(train_coco["images"]),
        "train_annotations": len(train_coco["annotations"]),
        "val_images": len(val_coco["images"]),
        "val_annotations": len(val_coco["annotations"]),
        "split_ratio": {"train": SPLIT_RATIO, "val": 1 - SPLIT_RATIO},
        "random_seed": RANDOM_SEED,
        "image_annotation_stats": stats,
        "conflict_counts": {key: len(value) for key, value in conflicts.items()},
        "anomaly_count": len(anomalies),
    }
    save_json(REPORT_DIR / "preparation_summary.json", summary_payload)

    text_summary = build_text_summary(
        coco_dict=coco_dict,
        train_coco=train_coco,
        val_coco=val_coco,
        stats=stats,
        split_map=split_map,
        conflicts=conflicts,
        anomalies=anomalies,
    )
    save_text_summary(REPORT_DIR / "preparation_summary.txt", text_summary)

    figure_paths = generate_figures(
        overall_counts=overall_counts,
        train_counts=train_counts,
        val_counts=val_counts,
        categories=categories,
    )

    print(text_summary)
    if figure_paths:
        print("")
        print("생성된 시각화 파일:")
        for path in figure_paths:
            print(f"- {path}")
    elif not MATPLOTLIB_AVAILABLE:
        print("")
        print("[WARN] matplotlib이 없어 시각화 생성에 실패했습니다.")


if __name__ == "__main__":
    main()
