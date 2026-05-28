"""
Dataset loader for CVUSA processed split dataset.

The dataset contains pre-processed 256x256 patches with the following structure:
- train/val/test/ground_rgb: RGB street view patches
- train/val/test/ground_gray: Grayscale street view patches
- train/val/test/overhead_polar: Polar coordinate satellite views

File naming: {panorama_id}_{patch_index}.jpg
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

import cv2
import numpy as np
from torch.utils.data import Dataset


class CVUSADataset(Dataset):
    """CVUSA dataset for satellite-guided street view colorization."""

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        use_segmap: bool = False,
        load_polar: bool = True,
        require_complete_panoramas: bool = True,
        polar_size: Tuple[int, int] = (256, 512),
    ):
        """
        Args:
            dataset_root: Path to CVUSA_processed_split directory
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
            use_segmap: Whether to load segmentation maps
            load_polar: Whether to load polar satellite views
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.use_segmap = use_segmap
        self.load_polar = load_polar
        self.require_complete_panoramas = require_complete_panoramas
        self.polar_size = polar_size

        # Validate paths
        self._validate_paths()

        # Build index
        self.samples = self._build_index()

    def _validate_paths(self):
        """Validate that required directories exist."""
        required_dirs = [
            self.dataset_root / self.split / "ground_rgb",
            self.dataset_root / self.split / "ground_gray",
        ]
        if self.load_polar:
            required_dirs.append(self.dataset_root / self.split / "overhead_polar")
        if self.use_segmap:
            required_dirs.append(self.dataset_root / self.split / "overhead_polar_seg")

        missing = [d for d in required_dirs if not d.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required directories: {[str(d) for d in missing]}"
            )

    def _build_index(self) -> List[Dict]:
        """Build index of samples with file paths.

        Actual naming conventions:
        - RGB: {panorama_id}_{patch_idx}.jpg
        - Gray: {panorama_id}_{patch_idx}_gray.jpg
        - Polar: {panorama_id}.jpg (without patch index)
        """
        rgb_dir = self.dataset_root / self.split / "ground_rgb"
        gray_dir = self.dataset_root / self.split / "ground_gray"
        polar_dir = self.dataset_root / self.split / "overhead_polar" if self.load_polar else None
        seg_dir = self.dataset_root / self.split / "overhead_polar_seg" if self.use_segmap else None

        samples = []
        missing_files = defaultdict(list)

        # Group files by panorama ID
        # RGB: {panorama_id}_{patch_idx}.jpg
        rgb_files = {f.stem: f for f in rgb_dir.glob("*.jpg") if f.is_file()}

        # Gray: {panorama_id}_{patch_idx}_gray.jpg
        # Extract {panorama_id}_{patch_idx} from the stem
        gray_files = {}
        for f in gray_dir.glob("*.jpg"):
            if f.is_file():
                stem = f.stem
                # Remove _gray suffix to get base ID
                if stem.endswith("_gray"):
                    base_id = stem[:-5]  # Remove "_gray"
                    gray_files[base_id] = f

        if self.load_polar:
            # Polar: {panorama_id}.jpg or {panorama_id}_{patch_idx}.jpg
            # First try to find panorama_id only, then use the first match if multiple exist
            polar_files = {}
            for f in polar_dir.glob("*.jpg"):
                if f.is_file():
                    stem = f.stem
                    # Extract panorama_id (remove patch index if present)
                    parts = stem.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        panorama_id = parts[0]
                    else:
                        panorama_id = stem

                    # Store panorama_id -> first matching file
                    if panorama_id not in polar_files:
                        polar_files[panorama_id] = f
        else:
            polar_files = {}

        if self.use_segmap:
            seg_files = {f.stem: f for f in seg_dir.glob("*.png") if f.is_file()}
        else:
            seg_files = {}

        for file_id, rgb_path in rgb_files.items():
            # Extract panorama ID (remove patch index)
            parts = file_id.rsplit("_", 1)
            if len(parts) != 2:
                continue
            panorama_id, patch_idx = parts[0], parts[1]

            # Check corresponding files
            # Gray file has _gray suffix
            gray_path = gray_files.get(f"{panorama_id}_{patch_idx}")
            polar_path = polar_files.get(panorama_id)  # Polar is shared across patches

            if not gray_path:
                missing_files['gray'].append(file_id)
                continue

            if self.load_polar and not polar_path:
                missing_files['polar'].append(file_id)
                continue

            sample = {
                "file_id": file_id,
                "panorama_id": panorama_id,
                "patch_idx": int(patch_idx) if patch_idx.isdigit() else 0,
                "rgb_path": str(rgb_path),
                "gray_path": str(gray_path),
                "polar_path": str(polar_path) if polar_path else None,
            }

            if self.use_segmap:
                seg_path = seg_files.get(panorama_id)
                sample["seg_path"] = str(seg_path) if seg_path else None

            samples.append(sample)

        if self.require_complete_panoramas:
            grouped = defaultdict(list)
            for sample in samples:
                grouped[sample["panorama_id"]].append(sample)
            samples = [
                sample
                for panorama_samples in grouped.values()
                if {s["patch_idx"] for s in panorama_samples} == {1, 2, 3, 4}
                for sample in panorama_samples
            ]

        samples.sort(key=lambda s: (s["panorama_id"], s["patch_idx"]))

        # Report missing files
        if any(missing_files.values()):
            print(f"[Dataset] Missing files in {self.split}:")
            for key, files in missing_files.items():
                if files:
                    print(f"  {key}: {len(files)} files")

        print(f"[Dataset] Loaded {len(samples)} samples from {self.split}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load images
        rgb = self._load_image(sample["rgb_path"])
        gray = self._load_image(sample["gray_path"], grayscale=True)

        polar = None
        if self.load_polar and sample["polar_path"]:
            polar = self._load_image(sample["polar_path"])
            polar = cv2.resize(
                polar, (self.polar_size[1], self.polar_size[0]), interpolation=cv2.INTER_LINEAR
            )

        seg = None
        if self.use_segmap and sample.get("seg_path"):
            seg = self._load_segmap(sample["seg_path"])

        result = {
            "rgb": rgb,
            "gray": gray,
            "polar": polar,
            "seg": seg,
            "file_id": sample["file_id"],
            "panorama_id": sample["panorama_id"],
            "patch_idx": sample["patch_idx"],
        }

        if self.transform:
            result = self.transform(result)

        return result

    def _load_image(self, path: str, grayscale: bool = False) -> np.ndarray:
        """Load image as numpy array, normalized to [0, 1]."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        if not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (256, 256) and grayscale:
            raise ValueError(f"Expected 256x256 patch, got {img.shape[:2]}: {path}")
        if img.shape[:2] != (256, 256) and "overhead_polar" not in path:
            raise ValueError(f"Expected 256x256 patch, got {img.shape[:2]}: {path}")
        return img.astype(np.float32) / 255.0

    def _load_segmap(self, path: str) -> np.ndarray:
        """Load segmentation map."""
        seg = cv2.imread(path, cv2.IMREAD_COLOR)
        if seg is None:
            return None
        return seg.astype(np.float32) / 255.0

    def get_dataset_info(self) -> Dict:
        """Return dataset statistics."""
        return {
            "split": self.split,
            "num_samples": len(self),
            "num_panoramas": len(set(s["panorama_id"] for s in self.samples)),
            "patches_per_panorama": self._get_patches_per_panorama(),
        }

    def _get_patches_per_panorama(self) -> Dict:
        """Get distribution of patches per panorama."""
        counts = defaultdict(int)
        for s in self.samples:
            counts[s["panorama_id"]] += 1
        return dict(sorted(counts.items()))


def create_dataset_report(dataset_root: str, output_path: str = None):
    """Generate a detailed dataset inspection report."""
    report = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "summary": {}
    }

    for split in ["train", "val", "test"]:
        try:
            dataset = CVUSADataset(dataset_root, split=split, load_polar=True)
            info = dataset.get_dataset_info()
            report["splits"][split] = info
        except Exception as e:
            report["splits"][split] = {"error": str(e)}

    # Summary
    total_samples = sum(s.get("num_samples", 0) for s in report["splits"].values())
    report["summary"] = {
        "total_samples": total_samples,
        "splits_with_errors": [k for k, v in report["splits"].items() if "error" in v]
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    # Test dataset
    dataset_root = r"C:\Users\31133\Desktop\dataset1\CVUSA_processed_split"
    report = create_dataset_report(dataset_root, "outputs/cache/dataset_report.json")
    print(json.dumps(report, indent=2))

    # Test loading a sample
    dataset = CVUSADataset(dataset_root, split="train")
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  RGB: {sample['rgb'].shape}")
    print(f"  Gray: {sample['gray'].shape}")
    print(f"  Polar: {sample['polar'].shape if sample['polar'] else 'N/A'}")
