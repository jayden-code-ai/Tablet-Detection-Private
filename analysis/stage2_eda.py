#!/usr/bin/env python3
"""
Stage 2 EDA pipeline for Tablet-Detection-with-Object-Detection.

This script aggregates COCO-style annotation JSON files, checks data integrity,
identifies duplicates and potential outliers, and generates summary reports
and matplotlib-based visualizations with Korean font support.

모든 경로는 절대 경로를 사용하며, 결과물은
`/mnt/nas/jayden_code/Tablet-Detection-Private/stage2_eda_artifacts` 하위에 저장됩니다.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - matplotlib은 외부 의존성
    plt = None  # type: ignore[assignment]
    font_manager = None  # type: ignore[assignment]
    rcParams = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

DATA_ROOT = Path("/mnt/nas/jayden_code/ai05-level1-project")
TRAIN_IMAGE_DIR = DATA_ROOT / "train_images"
TEST_IMAGE_DIR = DATA_ROOT / "test_images"
TRAIN_ANNOTATION_DIR = DATA_ROOT / "train_annotations"

OUTPUT_ROOT = Path("/mnt/nas/jayden_code/Tablet-Detection-Private/stage2_eda_artifacts")
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "reports"


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    width: int
    height: int
    image_id: int
    source_json: str


@dataclass(frozen=True)
class AnnotationRecord:
    file_name: str
    image_id: int
    category_id: int
    bbox: Tuple[float, float, float, float]
    width: int
    height: int
    source_json: str
    annotation_id: int


def ensure_output_directories() -> None:
    """결과물을 저장할 디렉터리를 생성한다."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def gather_annotation_paths(root: Path) -> List[Path]:
    """train_annotations 전체에서 JSON 경로를 수집한다."""
    return sorted(root.rglob("*.json"))


def hash_file(path: Path) -> str:
    """이미지 파일의 MD5 해시를 계산한다."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_korean_font() -> Optional[str]:
    """
    matplotlib에서 사용할 한글 폰트를 설정한다.

    - 시스템 폰트 탐색 시 한국어 지원 폰트(Nanum, Noto, Un*) 우선 적용
    - `rcParams['font.family']`에 등록 가능한 실제 패밀리명을 저장
    """
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
        "ungr",
    ]

    # 1) 시스템 폰트 경로에서 직접 탐색
    system_fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    seen_paths: List[str] = []
    for path in system_fonts:
        if path not in seen_paths:
            seen_paths.append(path)

    for keyword in preferred_keywords:
        for font_path in seen_paths:
            if keyword in font_path.lower():
                try:
                    font_manager.fontManager.addfont(font_path)
                    font_name = font_manager.FontProperties(fname=font_path).get_name()
                except Exception:
                    continue
                rcParams["font.family"] = font_name
                rcParams["axes.unicode_minus"] = False
                return font_name

    # 2) 이미 등록된 폰트 목록에서 검색
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for keyword in preferred_keywords:
        for font_name in available_fonts:
            if keyword in font_name.lower():
                rcParams["font.family"] = font_name
                rcParams["axes.unicode_minus"] = False
                return font_name

    rcParams["axes.unicode_minus"] = False
    return None


def collect_dataset(annotation_paths: Iterable[Path]) -> Tuple[
    Dict[str, ImageRecord],
    Dict[int, ImageRecord],
    List[AnnotationRecord],
    Dict[int, Dict[str, object]],
    Dict[str, List[ImageRecord]],
    Dict[int, List[ImageRecord]],
    Dict[int, List[str]],
    List[Dict[str, object]],
]:
    """
    JSON을 순회하며 이미지, 어노테이션, 카테고리 정보를 수집하고
    중복 및 누락 정보를 함께 반환한다.
    """
    image_by_file: Dict[str, ImageRecord] = {}
    image_by_id: Dict[int, ImageRecord] = {}
    annotation_records: List[AnnotationRecord] = []
    category_registry: Dict[int, Dict[str, object]] = {}

    image_name_sources: Dict[str, List[ImageRecord]] = defaultdict(list)
    image_id_sources: Dict[int, List[ImageRecord]] = defaultdict(list)
    annotation_id_sources: Dict[int, List[str]] = defaultdict(list)
    annotations_without_image: List[Dict[str, object]] = []

    for json_path in annotation_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            annotations_without_image.append(
                {
                    "issue": "json_decode_error",
                    "json_path": str(json_path),
                    "error": str(exc),
                }
            )
            continue

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        local_image_map = {img.get("id"): img for img in images if "id" in img}

        for img in images:
            file_name = img.get("file_name")
            image_id = img.get("id")
            width = img.get("width")
            height = img.get("height")

            if file_name is None or image_id is None or width is None or height is None:
                annotations_without_image.append(
                    {
                        "issue": "image_missing_fields",
                        "json_path": str(json_path),
                        "raw_image": img,
                    }
                )
                continue

            record = ImageRecord(
                file_name=file_name,
                width=int(width),
                height=int(height),
                image_id=int(image_id),
                source_json=str(json_path),
            )

            image_name_sources[file_name].append(record)
            image_id_sources[image_id].append(record)

            if file_name not in image_by_file:
                image_by_file[file_name] = record
            if image_id not in image_by_id:
                image_by_id[image_id] = record

        for category in categories:
            category_id = category.get("id")
            name = category.get("name")
            if category_id is None or name is None:
                continue

            info = category_registry.setdefault(
                int(category_id),
                {
                    "name": name,
                    "sources": [],
                    "supercategory": category.get("supercategory"),
                },
            )
            info["sources"].append(str(json_path))

            if info["name"] != name:
                info.setdefault("name_mismatch", set()).add(name)

        for ann in annotations:
            annotation_id = ann.get("id")
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")
            bbox = ann.get("bbox")

            if (
                annotation_id is None
                or image_id is None
                or category_id is None
                or not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
            ):
                annotations_without_image.append(
                    {
                        "issue": "annotation_missing_fields",
                        "json_path": str(json_path),
                        "raw_annotation": ann,
                    }
                )
                continue

            annotation_id_sources[int(annotation_id)].append(str(json_path))
            image_info = local_image_map.get(image_id)

            if image_info is None:
                annotations_without_image.append(
                    {
                        "issue": "annotation_missing_image_reference",
                        "json_path": str(json_path),
                        "annotation_id": annotation_id,
                        "image_id": image_id,
                    }
                )
                continue

            record = AnnotationRecord(
                file_name=image_info.get("file_name", ""),
                image_id=int(image_id),
                category_id=int(category_id),
                bbox=tuple(float(value) for value in bbox),
                width=int(image_info.get("width", 0)),
                height=int(image_info.get("height", 0)),
                source_json=str(json_path),
                annotation_id=int(annotation_id),
            )
            annotation_records.append(record)

    return (
        image_by_file,
        image_by_id,
        annotation_records,
        category_registry,
        image_name_sources,
        image_id_sources,
        annotation_id_sources,
        annotations_without_image,
    )


def compute_duplicates(source_map: Dict[str, List[ImageRecord]]) -> List[Dict[str, object]]:
    """중복된 이미지 파일명을 찾아 상세 정보를 반환한다."""
    duplicates = []
    for key, records in source_map.items():
        if len(records) > 1:
            duplicates.append(
                {
                    "key": key,
                    "occurrences": [
                        {
                            "image_id": record.image_id,
                            "width": record.width,
                            "height": record.height,
                            "source_json": record.source_json,
                        }
                        for record in records
                    ],
                }
            )
    return duplicates


def compute_annotation_duplicates(annotation_sources: Dict[int, List[str]]) -> List[Dict[str, object]]:
    """중복된 annotation id를 정리한다."""
    duplicates = []
    for key, sources in annotation_sources.items():
        if len(sources) > 1:
            duplicates.append({"annotation_id": key, "json_paths": sources})
    return duplicates


def compute_image_hash_duplicates(
    image_dir: Path, file_names: Iterable[str]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    이미지 파일 존재 여부와 해시 중복을 확인한다.
    반환값: (missing_images, hash_duplicates)
    """
    missing_images: List[Dict[str, object]] = []
    hash_map: Dict[str, List[str]] = defaultdict(list)

    for file_name in file_names:
        image_path = image_dir / file_name
        if not image_path.exists():
            missing_images.append(
                {"file_name": file_name, "expected_path": str(image_path)}
            )
            continue

        digest = hash_file(image_path)
        hash_map[digest].append(file_name)

    hash_duplicates = [
        {"hash": digest, "file_names": names}
        for digest, names in hash_map.items()
        if len(names) > 1
    ]
    return missing_images, hash_duplicates


def compute_bbox_statistics(annotations: List[AnnotationRecord]) -> Dict[str, object]:
    """바운딩 박스 기준 통계와 이상치를 계산한다."""
    area_ratios: List[float] = []
    width_ratios: List[float] = []
    height_ratios: List[float] = []
    aspect_ratios: List[float] = []
    invalid_bboxes: List[Dict[str, object]] = []

    for record in annotations:
        x, y, w, h = record.bbox
        if w <= 0 or h <= 0 or record.width <= 0 or record.height <= 0:
            invalid_bboxes.append(
                {
                    "file_name": record.file_name,
                    "bbox": record.bbox,
                    "image_id": record.image_id,
                    "source_json": record.source_json,
                }
            )
            continue

        image_area = record.width * record.height
        bbox_area = w * h
        area_ratio = bbox_area / image_area if image_area else 0.0
        area_ratios.append(area_ratio)
        width_ratios.append(w / record.width)
        height_ratios.append(h / record.height)
        aspect_ratios.append(w / h if h != 0 else math.inf)

    small_outliers = []
    large_outliers = []
    elongated_outliers = []

    for record, area_ratio, width_ratio, height_ratio, aspect_ratio in zip(
        annotations, area_ratios, width_ratios, height_ratios, aspect_ratios
    ):
        if area_ratio < 0.01:
            small_outliers.append(
                {
                    "file_name": record.file_name,
                    "area_ratio": area_ratio,
                    "bbox": record.bbox,
                    "source_json": record.source_json,
                    "annotation_id": record.annotation_id,
                }
            )
        if area_ratio > 0.8:
            large_outliers.append(
                {
                    "file_name": record.file_name,
                    "area_ratio": area_ratio,
                    "bbox": record.bbox,
                    "source_json": record.source_json,
                    "annotation_id": record.annotation_id,
                }
            )
        if aspect_ratio > 4 or aspect_ratio < 0.25:
            elongated_outliers.append(
                {
                    "file_name": record.file_name,
                    "aspect_ratio": aspect_ratio,
                    "bbox": record.bbox,
                    "source_json": record.source_json,
                    "annotation_id": record.annotation_id,
                }
            )

    area_stats = compute_basic_stats(area_ratios)
    width_stats = compute_basic_stats(width_ratios)
    height_stats = compute_basic_stats(height_ratios)
    aspect_stats = compute_basic_stats(aspect_ratios)

    return {
        "area_ratio_stats": area_stats,
        "width_ratio_stats": width_stats,
        "height_ratio_stats": height_stats,
        "aspect_ratio_stats": aspect_stats,
        "small_area_outliers": sorted(small_outliers, key=lambda x: x["area_ratio"])[:50],
        "large_area_outliers": sorted(
            large_outliers, key=lambda x: x["area_ratio"], reverse=True
        )[:50],
        "elongated_outliers": sorted(
            elongated_outliers,
            key=lambda x: abs(math.log(x["aspect_ratio"], 2)) if x["aspect_ratio"] else 0,
            reverse=True,
        )[:50],
        "invalid_bboxes": invalid_bboxes,
        "area_ratios": area_ratios,
        "width_ratios": width_ratios,
        "height_ratios": height_ratios,
        "aspect_ratios": aspect_ratios,
    }


def compute_basic_stats(values: List[float]) -> Dict[str, float]:
    """기본 통계 정보를 계산한다."""
    if not values:
        return {}

    stats: Dict[str, float] = {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
    if len(values) >= 4:
        q1, q2, q3 = statistics.quantiles(values, n=4)
        stats.update({"q1": q1, "q2": q2, "q3": q3})
    return stats


def compute_category_counts(annotations: List[AnnotationRecord]) -> Counter:
    """카테고리별 어노테이션 개수를 센다."""
    counter: Counter = Counter()
    for record in annotations:
        counter[record.category_id] += 1
    return counter


def build_text_summary(
    total_json: int,
    unique_images: int,
    annotation_count: int,
    unique_categories: int,
    duplicate_files: List[Dict[str, object]],
    duplicate_ids: List[Dict[str, object]],
    duplicate_annotations: List[Dict[str, object]],
    missing_files: List[Dict[str, object]],
    annotation_issues: List[Dict[str, object]],
    bbox_stats: Dict[str, object],
    category_counter: Counter,
    category_name_conflicts: List[Dict[str, object]],
) -> str:
    """사람이 읽기 쉬운 텍스트 요약을 만든다."""
    lines = [
        "=== Stage 2 EDA Summary ===",
        f"총 어노테이션 JSON 파일 수: {total_json}",
        f"고유 학습 이미지 수: {unique_images}",
        f"어노테이션 총 개수: {annotation_count}",
        f"고유 카테고리 수: {unique_categories}",
        "",
        f"중복 이미지 파일명 건수: {len(duplicate_files)}",
        f"중복 image_id 건수: {len(duplicate_ids)}",
        f"중복 annotation_id 건수: {len(duplicate_annotations)}",
        f"누락된 이미지 파일 건수: {len(missing_files)}",
        f"어노테이션 이슈 건수: {len(annotation_issues)}",
        f"카테고리 이름 불일치 건수: {len(category_name_conflicts)}",
    ]

    area_stats = bbox_stats.get("area_ratio_stats") or {}
    if area_stats:
        lines.append("")
        lines.append("바운딩 박스 면적 비율 통계:")
        for key in ["min", "q1", "median", "q3", "max", "mean"]:
            value = area_stats.get(key)
            if value is not None:
                lines.append(f"  {key}: {value:.6f}")

    elongated = bbox_stats.get("elongated_outliers") or []
    small = bbox_stats.get("small_area_outliers") or []
    large = bbox_stats.get("large_area_outliers") or []

    lines.append("")
    lines.append(f"세로/가로 비율 이상치 수: {len(elongated)}")
    lines.append(f"소형 면적 이상치 수(상위 50): {len(small)}")
    lines.append(f"대형 면적 이상치 수(상위 50): {len(large)}")

    lines.append("")
    lines.append("카테고리별 상위 10개 어노테이션 수:")
    for category_id, count in category_counter.most_common(10):
        lines.append(f"  category_id {category_id}: {count}")

    return "\n".join(lines)


def create_figures(
    bbox_stats: Dict[str, object],
    category_counter: Counter,
    category_registry: Dict[int, Dict[str, object]],
) -> List[str]:
    """matplotlib 시각화를 생성하고 파일 경로 목록을 반환한다."""
    figure_paths: List[str] = []

    if not MATPLOTLIB_AVAILABLE:
        return figure_paths

    configured_font = configure_korean_font()
    if configured_font:
        print(f"[INFO] matplotlib 한글 폰트 설정: {configured_font}")
    else:
        print("[WARN] 사용 가능한 한글 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")

    area_ratios: List[float] = bbox_stats.get("area_ratios", [])  # type: ignore[assignment]
    width_ratios: List[float] = bbox_stats.get("width_ratios", [])  # type: ignore[assignment]
    height_ratios: List[float] = bbox_stats.get("height_ratios", [])  # type: ignore[assignment]

    if area_ratios:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(area_ratios, bins=50, color="#1f77b4", edgecolor="#0a3a67")
        ax.set_title("바운딩 박스 면적 비율 분포")
        ax.set_xlabel("면적 비율 (bbox_area / image_area)")
        ax.set_ylabel("빈도")
        fig.tight_layout()
        output_path = FIGURE_DIR / "bbox_area_ratio_hist.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        figure_paths.append(str(output_path))

    if width_ratios and height_ratios:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(width_ratios, height_ratios, s=8, alpha=0.35, color="#ff7f0e")
        ax.set_title("바운딩 박스 폭/높이 비율 산점도")
        ax.set_xlabel("폭 비율 (bbox_w / image_w)")
        ax.set_ylabel("높이 비율 (bbox_h / image_h)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        output_path = FIGURE_DIR / "bbox_ratio_scatter.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        figure_paths.append(str(output_path))

    if category_counter:
        top_categories = category_counter.most_common(20)
        labels = [
            f"{category_registry.get(cid, {}).get('name', '') or cid}"
            for cid, _ in top_categories
        ]
        counts = [count for _, count in top_categories]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels[::-1], counts[::-1], color="#2ca02c")
        ax.set_title("카테고리별 어노테이션 수 (상위 20개)")
        ax.set_xlabel("어노테이션 수")
        fig.tight_layout()
        output_path = FIGURE_DIR / "top_category_counts.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        figure_paths.append(str(output_path))

    return figure_paths


def main() -> None:
    ensure_output_directories()

    annotation_paths = gather_annotation_paths(TRAIN_ANNOTATION_DIR)
    (
        image_by_file,
        image_by_id,
        annotation_records,
        category_registry,
        image_name_sources,
        image_id_sources,
        annotation_id_sources,
        annotation_issues,
    ) = collect_dataset(annotation_paths)

    duplicate_files = compute_duplicates(image_name_sources)
    duplicate_ids = compute_duplicates(image_id_sources)
    duplicate_annotations = compute_annotation_duplicates(annotation_id_sources)

    missing_files, hash_duplicates = compute_image_hash_duplicates(
        TRAIN_IMAGE_DIR, image_by_file.keys()
    )

    bbox_stats = compute_bbox_statistics(annotation_records)
    category_counter = compute_category_counts(annotation_records)
    category_name_conflicts = []
    category_registry_payload = []

    for category_id in sorted(category_registry):
        info = category_registry[category_id]
        mismatches = sorted(info.get("name_mismatch", [])) if info.get("name_mismatch") else []
        if mismatches:
            category_name_conflicts.append(
                {
                    "category_id": category_id,
                    "primary_name": info["name"],
                    "conflicting_names": mismatches,
                    "sources": info["sources"],
                }
            )
        payload = {
            "category_id": category_id,
            "name": info["name"],
            "supercategory": info.get("supercategory"),
            "sources": info["sources"],
        }
        if mismatches:
            payload["name_mismatch"] = mismatches
        category_registry_payload.append(payload)

    summary = {
        "total_annotation_json": len(annotation_paths),
        "unique_image_files": len(image_by_file),
        "unique_image_ids": len(image_by_id),
        "total_annotations": len(annotation_records),
        "unique_categories": len(category_registry),
        "duplicate_image_file_names": len(duplicate_files),
        "duplicate_image_ids": len(duplicate_ids),
        "duplicate_annotation_ids": len(duplicate_annotations),
        "missing_image_files": len(missing_files),
        "hash_duplicate_groups": len(hash_duplicates),
        "annotation_issues": len(annotation_issues),
        "category_name_conflict_count": len(category_name_conflicts),
        "bbox_stats": {
            key: value
            for key, value in bbox_stats.items()
            if key.endswith("_stats")
        },
        "matplotlib_available": MATPLOTLIB_AVAILABLE,
    }

    ensure_ascii = False
    (REPORT_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=ensure_ascii), encoding="utf-8"
    )
    (REPORT_DIR / "duplicate_image_files.json").write_text(
        json.dumps(duplicate_files, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    (REPORT_DIR / "duplicate_image_ids.json").write_text(
        json.dumps(duplicate_ids, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    (REPORT_DIR / "duplicate_annotation_ids.json").write_text(
        json.dumps(duplicate_annotations, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    (REPORT_DIR / "missing_image_files.json").write_text(
        json.dumps(missing_files, indent=2, ensure_ascii=ensure_ascii), encoding="utf-8"
    )
    (REPORT_DIR / "hash_duplicate_images.json").write_text(
        json.dumps(hash_duplicates, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    (REPORT_DIR / "annotation_issues.json").write_text(
        json.dumps(annotation_issues, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    (REPORT_DIR / "category_registry.json").write_text(
        json.dumps(category_registry_payload, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    (REPORT_DIR / "category_name_conflicts.json").write_text(
        json.dumps(category_name_conflicts, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    bbox_outlier_payload = {
        key: value
        for key, value in bbox_stats.items()
        if key
        in {
            "small_area_outliers",
            "large_area_outliers",
            "elongated_outliers",
            "invalid_bboxes",
        }
    }
    (REPORT_DIR / "bbox_outliers.json").write_text(
        json.dumps(bbox_outlier_payload, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )

    text_summary = build_text_summary(
        total_json=len(annotation_paths),
        unique_images=len(image_by_file),
        annotation_count=len(annotation_records),
        unique_categories=len(category_registry),
        duplicate_files=duplicate_files,
        duplicate_ids=duplicate_ids,
        duplicate_annotations=duplicate_annotations,
        missing_files=missing_files,
        annotation_issues=annotation_issues,
        bbox_stats=bbox_stats,
        category_counter=category_counter,
        category_name_conflicts=category_name_conflicts,
    )
    (REPORT_DIR / "eda_summary.txt").write_text(text_summary, encoding="utf-8")

    figure_paths = create_figures(bbox_stats, category_counter, category_registry)

    # 실행 결과를 표준 출력으로 간단히 안내한다.
    print(text_summary)
    if figure_paths:
        print("")
        print("생성된 시각화 파일:")
        for path in figure_paths:
            print(f"  - {path}")
    elif not MATPLOTLIB_AVAILABLE:
        print("")
        print(
            "[WARN] matplotlib 패키지가 설치되어 있지 않아 시각화를 생성하지 못했습니다. "
            "다음 명령으로 설치 후 다시 실행하세요:"
        )
        print("  python3 -m pip install --user matplotlib")


if __name__ == "__main__":
    main()
