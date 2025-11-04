#!/usr/bin/env python3
"""
Stage 5: YOLOv8 Large 학습 실행 스크립트.

- `configs/stage1_yolov8l.yaml`에서 하이퍼파라미터를 읽어들여 모델을 학습한다.
- 학습 결과는 `project` 경로(기본: /mnt/nas/jayden_code/Tablet-Detection-Private/stage5_yolov8l_runs)에 저장된다.
- GPU를 보유한 경우 자동으로 CUDA를 사용하도록 설정하며, 폴백 시 CPU로 전환된다.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from matplotlib import font_manager, rcParams
from ultralytics import YOLO
from ultralytics.data.dataset import (
    DATASET_CACHE_VERSION,
    YOLODataset,
    verify_image_label,
    save_dataset_cache_file,
)
from ultralytics.data.utils import get_hash
from ultralytics.utils import LOGGER, TQDM

from itertools import repeat

PROJECT_ROOT = Path("/mnt/nas/jayden_code/Tablet-Detection-Private")
MPL_CACHE_DIR = PROJECT_ROOT / "matplotlib_cache"
ULTRALYTICS_SETTINGS_DIR = PROJECT_ROOT / "ultralytics_settings"

MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ULTRALYTICS_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", str(ULTRALYTICS_SETTINGS_DIR))


def configure_korean_font() -> Optional[str]:
    """matplotlib 전역에 한글 폰트를 적용한다."""
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

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for keyword in preferred_keywords:
        for font_name in available_fonts:
            if keyword in font_name.lower().replace(" ", ""):
                rcParams["font.family"] = font_name
                rcParams["axes.unicode_minus"] = False
                return font_name

    rcParams["axes.unicode_minus"] = False
    return None


def cache_labels_sequential(self, path: Path = Path("./labels.cache")) -> dict:
    """
    YOLODataset.cache_labels 대체 함수.

    - ThreadPool 대신 단일 스레드로 라벨을 검증해 제한된 환경에서도 동작하도록 한다.
    """
    x = {"labels": []}
    nm = nf = ne = nc = 0
    msgs = []
    desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
    nkpt, ndim = self.data.get("kpt_shape", (0, 0))
    if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
        raise ValueError(
            "'kpt_shape' in data.yaml missing or incorrect. Should be a list with "
            "[number of keypoints, dims], e.g. 'kpt_shape: [17, 3]'"
        )

    iterable = zip(
        self.im_files,
        self.label_files,
        repeat(self.prefix),
        repeat(self.use_keypoints),
        repeat(len(self.data["names"])),
        repeat(nkpt),
        repeat(ndim),
        repeat(self.single_cls),
    )

    pbar = TQDM(iterable, desc=desc, total=len(self.im_files))
    for args in pbar:
        (
            im_file,
            lb,
            shape,
            segments,
            keypoint,
            nm_f,
            nf_f,
            ne_f,
            nc_f,
            msg,
        ) = verify_image_label(args)

        nm += nm_f
        nf += nf_f
        ne += ne_f
        nc += nc_f

        if im_file:
            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": shape,
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": segments,
                    "keypoints": keypoint,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if msg:
            msgs.append(msg)
        pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
    pbar.close()

    if msgs:
        LOGGER.info("\n".join(msgs))
    if nf == 0:
        LOGGER.warning(f"{self.prefix}No labels found in {path}. https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/")

    x["hash"] = get_hash(self.label_files + self.im_files)
    x["results"] = nf, nm, ne, nc, len(self.im_files)
    x["msgs"] = msgs
    save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
    return x


# ThreadPool을 사용하지 않도록 YOLODataset의 cache_labels 메서드를 교체한다.
YOLODataset.cache_labels = cache_labels_sequential

DEFAULT_CONFIG = Path("/mnt/nas/jayden_code/Tablet-Detection-Private/configs/stage1_yolov8l.yaml")


def load_config(path: Path) -> Dict[str, Any]:
    """YAML 설정 파일을 읽어 딕셔너리로 반환한다."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} 에서 유효한 딕셔너리를 로드하지 못했습니다.")
    return config


def resolve_device(device_value: Any) -> str:
    """
    설정된 device 값을 기반으로 최종 디바이스를 결정한다.
    GPU 사용이 불가하면 CPU로 자동 폴백한다.
    """
    if device_value is None:
        device_value = "auto"

    if isinstance(device_value, str):
        normalized = device_value.lower()
    else:
        normalized = device_value

    if normalized in ("auto", "gpu"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if isinstance(device_value, int):
        if torch.cuda.is_available():
            return str(device_value)
        LOGGER.warning("요청한 GPU 디바이스를 사용할 수 없어 CPU로 전환합니다.")
        return "cpu"

    if isinstance(device_value, str):
        if normalized.startswith("cuda") or normalized.isdigit():
            if torch.cuda.is_available():
                return device_value
            LOGGER.warning("요청한 GPU 디바이스를 사용할 수 없어 CPU로 전환합니다.")
            return "cpu"
        if normalized == "cpu":
            return "cpu"

    LOGGER.warning(f"알 수 없는 device 설정 '{device_value}' 입니다. 자동으로 디바이스를 선택합니다.")
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def train_from_config(config_path: Path) -> Dict[str, Any]:
    """YOLOv8 훈련을 수행하고 Ultralytics의 반환값(dict)을 돌려준다."""
    config = load_config(config_path)

    model_path = config.pop("model", None)
    if not model_path:
        raise ValueError("설정 파일에 'model' 키가 필요합니다.")

    resolved_device = resolve_device(config.get("device"))
    if resolved_device != config.get("device"):
        config["device"] = resolved_device
    LOGGER.info(f"최종 학습 디바이스: {config['device']}")

    font_name = configure_korean_font()
    if font_name:
        LOGGER.info(f"matplotlib 한글 폰트 설정: {font_name}")
    else:
        LOGGER.warning("한글 폰트를 찾지 못했습니다. 시각화에서 글자가 깨질 수 있습니다.")

    model = YOLO(model_path)
    results = model.train(**config)

    # Ultralytics의 train 메서드는 내부적으로 dict-like 객체를 반환한다.
    # JSON 직렬화 가능한 객체로 변환한다.
    serializable = {
        "metrics": getattr(results, "metrics", None),
        "save_dir": str(getattr(results, "save_dir", "")),
        "train_args": config,
    }
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 YOLOv8 Large 학습 스크립트")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="학습 설정 YAML 경로 (기본: configs/stage1_yolov8l.yaml)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/mnt/nas/jayden_code/Tablet-Detection-Private/stage5_yolov8l_runs/last_summary.json"),
        help="학습 결과 메타데이터를 저장할 경로",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    summary_path = args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] YOLOv8 Large 학습을 시작합니다. 설정 파일: {config_path}")
    results = train_from_config(config_path)

    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] 학습 결과 요약을 {summary_path} 에 저장했습니다.")


if __name__ == "__main__":
    main()
