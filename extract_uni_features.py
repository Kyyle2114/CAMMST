from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import UNI
from utils.data import create_uni_inference_loader, list_slides_from_wsi

# --- Configuration ---
OUTPUT_ROOT = Path("uni_features")
NUM_WORKERS = 4


def ensure_output_dir(root: Path, dataset_name: str, slide: str) -> Path:
    """Create output directory for a slide and return its path."""
    slide_dir = root / dataset_name / slide
    slide_dir.mkdir(parents=True, exist_ok=True)
    return slide_dir


def save_features(output_dir: Path, features: torch.Tensor) -> None:
    """Save UNI features as .npy."""
    np.save(output_dir / "uni_features.npy", features.cpu().numpy())


def main(dataset: str) -> None:
    """Entrypoint for extracting UNI features."""
    data_path = Path(f'./data/{dataset}')
    slides = list_slides_from_wsi(data_path)
    output_root = OUTPUT_ROOT
    dataset_name = dataset

    # minimal config needed by create_uni_inference_loader
    config = {"data": {"dataset_path": str(data_path)}}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader: DataLoader = create_uni_inference_loader(
        config=config,
        slides=slides,
        batch_size=1,
        num_workers=NUM_WORKERS,
    )

    uni = UNI()
    uni.to(device)

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "total_slides": len(slides),
        "success": [],
        "fail": [],
    }

    for batch in dataloader:
        slide_name: str = batch["slide_name"]
        try:
            images: torch.Tensor = batch["images"].to(device, non_blocking=True)

            with torch.inference_mode():
                feats = uni(images)  # (num_spots, embed_dim)

            slide_dir = ensure_output_dir(output_root, dataset_name, slide_name)
            save_features(slide_dir, feats)
            meta = {
                "dataset": dataset_name,
                "slide_name": slide_name,
                "feature_file": "uni_features.npy",
                "shape": list(feats.shape),
                "dtype": str(feats.dtype).replace("torch.", ""),
                "embed_dim": feats.shape[-1],
                "num_spots": feats.shape[0],
                "model_name": "hf-hub:MahmoodLab/UNI2-h",
                "pretrained": True,
                "device": str(device),
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "data_path": str(data_path),
                "wsi_dir": str(data_path / "wsi"),
                "patch_size_raw": [256, 256],
                "wsi_extension": ".jpg",
            }
            with (slide_dir / "meta.json").open("w") as mf:
                json.dump(meta, mf, indent=2)

            summary["success"].append(
                {
                    "slide_name": slide_name,
                    "num_spots": feats.shape[0],
                    "embed_dim": feats.shape[-1],
                    "feature_file": str(slide_dir / "uni_features.npy"),
                }
            )
            print(f"[OK] {slide_name} -> {slide_dir}")
        except Exception as exc:  
            err_msg = "".join(traceback.format_exception_only(exc.__class__, exc)).strip()
            summary["fail"].append({"slide_name": slide_name, "error": err_msg})
            print(f"[FAIL] {slide_name} -> {err_msg}")

    # write dataset-level summary
    summary["num_success"] = len(summary["success"])
    summary["num_fail"] = len(summary["fail"])
    summary["success_rate"] = (
        summary["num_success"] / summary["total_slides"] if summary["total_slides"] else 0.0
    )
    with (dataset_dir / "dataset_meta.json").open("w") as sf:
        json.dump(summary, sf, indent=2)


if __name__ == "__main__":
    for dataset in ["her2st", "skin", "stnet"]:
        main(dataset=dataset)
        print(f"Finished processing {dataset} \n")
    print("All datasets processed \n")