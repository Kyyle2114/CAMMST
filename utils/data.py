import numpy as np
import pandas as pd
import skimage.io
import PIL
from sklearn.model_selection import KFold
from pathlib import Path
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Any, Dict, List, Optional, Sequence, Tuple
from scipy.spatial import cKDTree

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .misc import seed_worker

PIL.Image.MAX_IMAGE_PIXELS = None


# --- UNI inference: extract patches from WSI ---

def list_slides_from_wsi(data_path: Path) -> List[str]:
    """List slide IDs from jpg files under data_path/wsi."""
    wsi_dir = data_path / "wsi"
    if not wsi_dir.exists():
        raise FileNotFoundError(f"WSI directory not found: {wsi_dir}")
    slides = sorted([p.stem for p in wsi_dir.glob("*.jpg")])
    if not slides:
        raise FileNotFoundError(f"No .jpg slides found in {wsi_dir}")
    return slides


class UNIInferenceDataset(Dataset):
    """Dataset for running UNI foundation model inference slide-by-slide."""

    def __init__(
        self,
        slides: Sequence[str],
        config: Dict[str, Any],
        transform: transforms.Compose,
    ) -> None:
        self.slides = list(slides)
        self.config = config
        self.transform = transform
        self.slide_samples: List[Dict[str, Any]] = []

        data_path = config["data"]["dataset_path"]

        for slide in self.slides:
            barcodes_path = Path(data_path) / "barcodes" / f"{slide}.csv"
            tissue_path = Path(data_path) / "tissue_positions" / f"{slide}.csv"

            if not barcodes_path.exists():
                raise FileNotFoundError(f"barcodes file not found: {barcodes_path}")
            if not tissue_path.exists():
                raise FileNotFoundError(f"tissue_positions file not found: {tissue_path}")

            barcodes = pd.read_csv(barcodes_path, header=None)[0].values
            tissue_positions = pd.read_csv(tissue_path, index_col=0)
            tissue_positions = tissue_positions[tissue_positions["in_tissue"] == 1]

            if len(tissue_positions) != len(barcodes):
                msg = (
                    f"mismatched spot counts for slide {slide}: "
                    f"barcodes={len(barcodes)}, tissue={len(tissue_positions)}"
                )
                raise ValueError(msg)

            wsi_name: Optional[str] = None
            if (Path(data_path) / "wsi" / f"{slide}.tif").exists():
                wsi_name = f"{slide}.tif"
            elif (Path(data_path) / "wsi" / f"{slide}.tiff").exists():
                wsi_name = f"{slide}.tiff"
            elif (Path(data_path) / "wsi" / f"{slide}.svs").exists():
                wsi_name = f"{slide}.svs"
            elif (Path(data_path) / "wsi" / f"{slide}.jpg").exists():
                wsi_name = f"{slide}.jpg"
            if wsi_name is None:
                raise FileNotFoundError(
                    f"WSI file not found for slide {slide} with supported extensions (.tif/.tiff/.svs/.jpg)"
                )

            wsi = skimage.io.imread(Path(data_path) / "wsi" / wsi_name)

            x_coords = tissue_positions["pxl_col_in_fullres"].values
            y_coords = tissue_positions["pxl_row_in_fullres"].values

            patches: List[np.ndarray] = []
            coords: List[Tuple[int, int]] = []
            barcodes_list: List[str] = []

            for local_idx, (x, y, barcode) in enumerate(zip(x_coords, y_coords, barcodes)):
                x, y = round(x), round(y)

                if x < 128:
                    x = 128
                if y < 128:
                    y = 128
                if x > wsi.shape[1] - 128:
                    x = wsi.shape[1] - 128
                if y > wsi.shape[0] - 128:
                    y = wsi.shape[0] - 128

                patch = wsi[y - 128 : y + 128, x - 128 : x + 128, :3]
                if patch.shape != (256, 256, 3):
                    raise ValueError(
                        f"Invalid patch shape {patch.shape} for slide {slide} at ({x}, {y}), wsi={wsi.shape}"
                    )

                patches.append(patch.astype(np.uint8))
                coords.append((x, y))
                barcodes_list.append(str(barcode))

            self.slide_samples.append(
                {
                    "slide_name": slide,
                    "patches": patches,
                    "coords": coords,
                    "barcodes": barcodes_list,
                }
            )

    def __len__(self) -> int:
        return len(self.slide_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        slide_sample = self.slide_samples[idx]
        transformed = [
            self.transform(PIL.Image.fromarray(img)) if self.transform else img for img in slide_sample["patches"]
        ]
        images_tensor = torch.stack(transformed, dim=0)
        spot_indices = list(range(len(transformed)))
        return {
            "images": images_tensor,  # (num_spots, C, H, W)
            "slide_name": slide_sample["slide_name"],
            "spot_idx": spot_indices,
            "coords": slide_sample["coords"],
            "barcodes": slide_sample["barcodes"],
        }


def create_uni_inference_loader(
    config: Dict[str, Any],
    slides: Sequence[str] | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:
    """Create dataloader for running UNI foundation model inference over all spots."""
    slides_list = list(slides) if slides is not None else sorted(
        list_slides_from_wsi(Path(config["data"]["dataset_path"]))
    )

    # use timm's create_transform with UNI2-h pretrained config
    model_name = "hf-hub:MahmoodLab/UNI2-h"
    # create a temporary model to get pretrained config
    temp_model = timm.create_model(model_name, pretrained=False, num_classes=0)
    fm_transform = create_transform(**resolve_data_config(temp_model.pretrained_cfg, model=temp_model))
    del temp_model  # free memory

    dataset = UNIInferenceDataset(slides=slides_list, config=config, transform=fm_transform)

    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # batch_size is intended to be slide-level; when batch_size=1, return the single slide dict
        if len(batch) == 1:
            return batch[0]
        # if batch_size > 1, keep lists for slide-level aggregation
        return {
            "images": [b["images"] for b in batch],
            "slide_name": [b["slide_name"] for b in batch],
            "spot_idx": [b["spot_idx"] for b in batch],
            "coords": [b["coords"] for b in batch],
            "barcodes": [b["barcodes"] for b in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


# --- Cached feature loader: read saved .npy embeddings ---

def split_slides(config: Dict[str, Any], fold: int) -> Tuple[List[str], List[str]]:
    # read slides directly from WSI directory instead of CSV
    data_path = Path(config["data"]["dataset_path"])
    slides = list_slides_from_wsi(data_path)
    n_splits = int(config["data"]["folds"])
    seed = int(config["general"]["seed"])
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if fold < 0 or fold >= n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}], got {fold}")

    slides_arr = np.asarray(slides)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = list(kf.split(slides_arr))[fold]
    train_slides = slides_arr[train_idx].tolist()
    val_slides = slides_arr[val_idx].tolist()
    return train_slides, val_slides


class UNIFeatureDataset(Dataset):
    """Dataset that loads precomputed UNI features and matching counts/barcodes per slide."""

    def __init__(
        self,
        slides: Sequence[str],
        config: Dict[str, Any],
        k_neighbors: int = 8,
    ) -> None:
        self.slides = list(slides)
        self.config = config
        self.k_neighbors = k_neighbors

        data_path = Path(config["data"]["dataset_path"])
        dataset_name = config["data"].get("dataset_name", "stnet")
        self.feature_root = Path(config["data"].get("feature_path", "./uni_features")) / dataset_name

        self.counts_root = data_path / "counts_spcs_to_8n"
        self.barcodes_root = data_path / "barcodes"

        # caches to avoid repeated file I/O
        self.bio_score_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.barcodes_cache: Dict[str, np.ndarray] = {}
        self.coords_cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        slide = self.slides[index]
        features = self._load_features(slide)
        counts = self._load_counts(slide)
        barcodes = self._get_or_load_barcodes(slide)
        coords_tensor = self._get_or_load_coords(slide)

        if len(barcodes) != counts.shape[0]:
            raise ValueError(
                f"Spot count mismatch for {slide}: barcodes={len(barcodes)}, counts={counts.shape[0]}"
            )
        if len(features) != counts.shape[0]:
            raise ValueError(
                f"Feature/Count mismatch for {slide}: features={len(features)}, counts={counts.shape[0]}"
            )
        if coords_tensor.shape[0] != counts.shape[0]:
            raise ValueError(
                f"Coords mismatch for {slide}: coords={coords_tensor.shape[0]}, counts={counts.shape[0]}"
            )

        nonzero_mask = counts.sum(axis=1) > 0
        features = features[nonzero_mask]
        counts = counts[nonzero_mask]
        barcodes = barcodes[nonzero_mask]
        coords_tensor = coords_tensor[nonzero_mask]

        barcodes_list = barcodes.astype(str).tolist()

        # get cached bio-salience scores or compute them
        global_bio_salience_score, local_bio_salience_score = self._get_or_compute_bio_scores(
            slide, counts, coords_tensor
        )

        features_tensor = torch.tensor(features, dtype=torch.float32)
        counts_tensor = torch.tensor(counts, dtype=torch.float32)

        return {
            "slide_name": slide,
            "features": features_tensor,
            "gt_expressions": counts_tensor,
            "barcodes": barcodes_list,
            "coords": coords_tensor,
            "global_bio_salience_score": global_bio_salience_score,  # (N,) - global score
            "local_bio_salience_score": local_bio_salience_score,  # (N,) - local score
        }

    def _load_features(self, slide: str) -> np.ndarray:
        # expect uni_features/<dataset>/<slide>/uni_features.npy by default
        feature_file = self.feature_root / slide / "uni_features.npy"
        if not feature_file.exists():
            raise FileNotFoundError(f"FM feature file not found: {feature_file}")
        return np.load(feature_file)

    def _load_counts(self, slide: str) -> np.ndarray:
        counts_file = self.counts_root / f"{slide}.npy"
        if not counts_file.exists():
            raise FileNotFoundError(f"Counts file not found: {counts_file}")
        return np.load(counts_file)

    def _load_barcodes(self, slide: str) -> Optional[np.ndarray]:
        barcode_file = self.barcodes_root / f"{slide}.csv"
        if not barcode_file.exists():
            raise FileNotFoundError(f"Barcodes file not found: {barcode_file}")
        df = pd.read_csv(barcode_file, header=None)
        return df[0].to_numpy(dtype=str)

    def _get_or_compute_bio_scores(
        self, slide: str, counts: np.ndarray, coords_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached bio-salience scores or compute and cache them for a slide."""
        # check cache first
        if slide in self.bio_score_cache:
            return self.bio_score_cache[slide]

        # compute bio-salience scores
        global_score, local_score = self._compute_bio_scores(counts, coords_tensor)

        # cache the results
        self.bio_score_cache[slide] = (global_score, local_score)

        return global_score, local_score

    def _compute_bio_scores(
        self, counts: np.ndarray, coords_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute global and local bio-salience scores for given counts and coordinates."""
        # --- Bio-Salience Score Calculation ---
        # compute bio-salience score (deviation from slide-specific mean)
        # used as target for Bio-Salience Guided Sampling Loss

        # gene-wise Z-score normalization
        counts_mean = counts.mean(axis=0, keepdims=True)  # (1, G)
        counts_std = counts.std(axis=0, keepdims=True)
        median_std = np.percentile(counts_std, 50)
        reg_term = 0.1 * median_std + 1e-8
        
        counts_normalized = (counts - counts_mean) / (counts_std + reg_term)  # (N, G)

        # --- Global bio-salience score calculation ---
        # slide-specific mean vector
        slide_mean = counts_normalized.mean(axis=0, keepdims=True)  # (1, G)

        # calculate deviation (L2 norm)
        deviation = np.linalg.norm(counts_normalized - slide_mean, axis=1)  # (N,)

        # --- Local bio-salience score calculation ---
        # compute local bio-salience score (deviation from k-nearest neighbors mean)
        # use spatial coordinates to find local neighborhood

        # build KDTree for efficient nearest neighbor search
        coords_np = coords_tensor.numpy()  # (N, 2)
        kdtree = cKDTree(coords_np)

        # adjust k_neighbors if it exceeds available spots
        effective_k = min(self.k_neighbors, len(coords_np) - 1)

        if effective_k > 0:
            # find k nearest neighbors for each spot (excluding self)
            distances, indices = kdtree.query(coords_np, k=effective_k + 1)  # +1 to exclude self

            # remove self from neighbors (first column is always self with distance 0)
            neighbor_indices = indices[:, 1:]  # (N, k)

            # compute local mean for each spot using neighbor expressions
            neighbor_exprs = counts_normalized[neighbor_indices]
            local_means = neighbor_exprs.mean(axis=1)

            # calculate local deviation (L2 norm from local neighborhood mean)
            local_deviation = np.linalg.norm(counts_normalized - local_means, axis=1)  # (N,)
        else:
            # fallback to global score if no neighbors available
            local_deviation = deviation
        
        # --- Sigmoid Gating --- 
        total_counts = counts.sum(axis=1) # (N,)
        
        # threshold for sigmoid gating (20th percentile of total counts)
        gate_center = np.percentile(total_counts, 20)
        sigmoid_slope = 0.05
        
        # sigmoid function
        weights = 1 / (1 + np.exp(-sigmoid_slope * (total_counts - gate_center)))
        
        # apply weights
        local_deviation = local_deviation * weights
        deviation = deviation * weights
        
        global_bio_salience_score = torch.tensor(deviation, dtype=torch.float32)
        local_bio_salience_score = torch.tensor(local_deviation, dtype=torch.float32)
        
        return global_bio_salience_score, local_bio_salience_score

    def _get_or_load_barcodes(self, slide: str) -> np.ndarray:
        """Get cached barcodes or load and cache them for a slide."""
        if slide in self.barcodes_cache:
            return self.barcodes_cache[slide]

        barcodes = self._load_barcodes(slide)
        self.barcodes_cache[slide] = barcodes
        return barcodes

    def _get_or_load_coords(self, slide: str) -> torch.Tensor:
        """Get cached coordinates or load and cache them for a slide."""
        if slide in self.coords_cache:
            return self.coords_cache[slide]

        coords_tensor = self._load_coords(slide)
        self.coords_cache[slide] = coords_tensor
        return coords_tensor

    def clear_caches(self) -> None:
        """Clear all caches (bio-scores, barcodes, coords) to free memory."""
        self.bio_score_cache.clear()
        self.barcodes_cache.clear()
        self.coords_cache.clear()

    def _load_coords(self, slide: str) -> Optional[torch.Tensor]:
        """Load WSI pixel coordinates for ALiBi positional embedding."""
        tissue_positions_file = Path(self.config["data"]["dataset_path"]) / "tissue_positions" / f"{slide}.csv"
        if not tissue_positions_file.exists():
            raise FileNotFoundError(f"Tissue positions file not found: {tissue_positions_file}")

        df = pd.read_csv(tissue_positions_file, index_col=0)
        df = df[df["in_tissue"] == 1]  # only tissue spots

        # extract pixel coordinates
        coords = df[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(np.float32)

        # normalize coordinates to prevent large values in ALiBi calculations
        # WSI coordinates are typically in thousands to tens of thousands of pixels
        # normalize to approximately [-1, 1] range for stable attention computations
        coords_min = coords.min(axis=0, keepdims=True)
        coords_max = coords.max(axis=0, keepdims=True)
        coords_range = coords_max - coords_min

        # avoid division by zero
        coords_range = np.where(coords_range == 0, 1.0, coords_range)

        # min-max normalization to [0, 1], then shift to [-0.5, 0.5], then scale to [-1, 1]
        coords_normalized = (coords - coords_min) / coords_range  # [0, 1]
        coords_normalized = coords_normalized - 0.5  # [-0.5, 0.5]
        coords_normalized = coords_normalized * 2.0  # [-1, 1]

        return torch.tensor(coords_normalized, dtype=torch.float32)


def create_feature_dataloaders(
    train_slides: Sequence[str],
    val_slides: Sequence[str],
    config: Dict[str, Any],
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    train_dataset = UNIFeatureDataset(train_slides, config)
    val_dataset = UNIFeatureDataset(val_slides, config)
    
    dataloader_kwargs = {
        "batch_size": 1,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    
    return {
        "train": DataLoader(train_dataset, shuffle=True, **dataloader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **dataloader_kwargs),
    }


def feature_dataset_sizes(train_slides: Sequence[str], val_slides: Sequence[str]) -> Dict[str, int]:
    return {"train": len(train_slides), "val": len(val_slides)}