"""
Dataset loader for PolypGen C1-C6 dataset with Fed-FGS support:
- Standard SAM RGB normalization and 1024x1024 resolution.
- Discrete Fourier Transform Hard-Thresholding (DFT-HT) frequency space augmentation.
- Amplitude pool construction across training clients (strictly excluding held-out Center 6).
- Metadata extraction for small vs large polyp stratification and difficulty-aware gradient scaling.
"""
import os
import glob
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from feddg_dftht import extract_amp_spectrum, freq_space_interpolation

# Map Flower client IDs ("0"..."5") to PolypGen clients (C1-C6)
CLIENT_MAP = {
    "0": "C1",
    "1": "C2",
    "2": "C3",
    "3": "C4",
    "4": "C5",
    "5": "C6",
}

DEFAULT_DATA_ROOT = "/home/omanchat-1/Documents/fed/fed_sam/finetune-SAM/polyp_db_dataset/data"
DEFAULT_META_ROOT = "/home/omanchat-1/Documents/fed/fed_sam/finetune-SAM/polyp_db_dataset/metadata"


def load_polyp_metadata(metadata_dir: str = DEFAULT_META_ROOT) -> Dict[str, dict]:
    """
    Load all my_dataDetails_C*.csv files into a fast O(1) lookup dictionary.
    Keys are both basename (e.g. 'EndoCV2021_C6_0100001.jpg') and full relative paths.
    """
    metadata_lookup = {}
    if not os.path.exists(metadata_dir):
        return metadata_lookup

    pattern = os.path.join(metadata_dir, "my_dataDetails_C*.csv")
    for csv_file in glob.glob(pattern):
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                img_name = str(row["img_name"]).strip()
                base_name = os.path.basename(img_name)
                is_small = bool(row["small_polyp"])
                inv_area = float(row["inv_area"]) if "inv_area" in row and pd.notnull(row["inv_area"]) else 0.0

                info = {"small_polyp": is_small, "inv_area": inv_area}
                metadata_lookup[img_name] = info
                metadata_lookup[base_name] = info
        except Exception as e:
            print(f"[Metadata] Warning loading {csv_file}: {e}")

    return metadata_lookup


class PolypDataset(Dataset):
    """
    PyTorch Dataset for PolypGen client partition with Fed-FGS DFT-HT support.
    """
    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        client_id: str = "0",
        mode: str = "train",
        image_size: int = 1024,
        target_amp_pool: Optional[List[torch.Tensor]] = None,
        dftht_prob: float = 0.3,
        dftht_L: float = 0.1,
        dftht_alpha: float = 0.05,
        metadata_dict: Optional[Dict[str, dict]] = None,
        split_type: str = "fold",
        fold_nr: int = 1,
        splits_root: Optional[str] = None,
    ):
        super(PolypDataset, self).__init__()
        self.data_root = data_root or DEFAULT_DATA_ROOT
        self.client_id = str(client_id)
        self.client_name = CLIENT_MAP.get(self.client_id, f"C{int(client_id) + 1}")
        self.mode = mode
        self.image_size = image_size
        self.target_amp_pool = target_amp_pool or []
        self.dftht_prob = dftht_prob
        self.dftht_L = dftht_L
        self.dftht_alpha = dftht_alpha
        self.metadata_dict = metadata_dict if metadata_dict is not None else {}
        self.split_type = split_type
        self.fold_nr = fold_nr
        self.splits_root = splits_root or "fed-dftht-main/data_polypgen/splits/fold_splits"

        self.samples = []

        if self.split_type == "fold" and os.path.exists(self.splits_root):
            c_nr = int(self.client_id) + 1
            if c_nr == 6:
                # Center 6 is the held-out unseen center.
                # In paper, out_test evaluates all c6 splits (88 samples)
                c6_dir = "fed-dftht-main/data_polypgen/splits/center_splits/c6"
                if self.mode in ["test", "out_test"]:
                    split_files = ["train_split.csv", "val_split.csv", "in_test_split.csv"]
                elif self.mode == "val":
                    split_files = ["val_split.csv"]
                else:
                    split_files = ["train_split.csv"]

                csv_paths = [os.path.join(c6_dir, sf) for sf in split_files]
            else:
                fold_dir = os.path.join(self.splits_root, f"c{c_nr}/fold{self.fold_nr}")
                if self.mode == "train":
                    split_files = ["train_split.csv"]
                elif self.mode == "val":
                    split_files = ["val_split.csv"]
                else:
                    split_files = ["in_test_split.csv"]

                csv_paths = [os.path.join(fold_dir, sf) for sf in split_files]

            for cp in csv_paths:
                if not os.path.exists(cp):
                    continue
                df = pd.read_csv(cp)
                for _, row in df.iterrows():
                    img_rel = str(row["image"]).strip()
                    mask_rel = str(row["mask"]).strip()
                    if img_rel.startswith("data/"):
                        img_rel = img_rel[len("data/"):]
                    if mask_rel.startswith("data/"):
                        mask_rel = mask_rel[len("data/"):]

                    img_path = os.path.join(self.data_root, img_rel)
                    mask_path = os.path.join(self.data_root, mask_rel)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
        else:
            # Fallback to direct client CSVs
            self.client_dir = os.path.join(self.data_root, f"data_{self.client_name}")
            self.csv_path = os.path.join(self.client_dir, f"{mode}.csv")
            if os.path.exists(self.csv_path):
                with open(self.csv_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) != 2:
                        continue
                    img_rel = parts[0].strip().lstrip("/")
                    mask_rel = parts[1].strip().lstrip("/")
                    img_path = os.path.join(self.client_dir, img_rel)
                    mask_path = os.path.join(self.client_dir, mask_rel)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        base_name = os.path.basename(img_path)

        # Load RGB image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_tensor = self.normalize(image_tensor)

        # Apply DFT-HT frequency augmentation stochastically during training
        if (
            self.mode == "train"
            and len(self.target_amp_pool) > 0
            and torch.rand(1).item() < self.dftht_prob
        ):
            target_idx = np.random.randint(len(self.target_amp_pool))
            amp_target = self.target_amp_pool[target_idx].to(image_tensor.device)
            ratio = float(np.random.uniform(0.0, 1.0))
            image_tensor = freq_space_interpolation(
                image_tensor,
                amp_target,
                L=self.dftht_L,
                ratio=ratio,
                alpha=self.dftht_alpha,
            )

        # Load binary mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_np = (np.array(mask, dtype=np.uint8) > 127).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np).long().unsqueeze(0)  # (1, H, W)

        # Extract polyp size details for Difficulty Gradient Scaling
        meta_info = self.metadata_dict.get(base_name, None)
        if meta_info is not None:
            is_small = meta_info["small_polyp"]
            inv_area = meta_info["inv_area"]
        else:
            polyp_pixels = int(mask_tensor.sum().item())
            total_pixels = int(mask_tensor.numel())
            if polyp_pixels > 0:
                inv_area = float(total_pixels) / float(polyp_pixels)
                # In colonoscopy, small polyps typically occupy < 2% of field of view
                is_small = (polyp_pixels / total_pixels) < 0.02
            else:
                inv_area = 0.0
                is_small = False

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": base_name,
            "image_path": img_path,
            "small_polyp": torch.tensor(1 if is_small else 0, dtype=torch.long),
            "inv_area": torch.tensor(inv_area, dtype=torch.float32),
        }


def build_amplitude_pools_for_clients(
    client_ids: List[str],
    out_center: str = "5",  # Center 6 (cid="5") is strictly held out
    data_root: str = DEFAULT_DATA_ROOT,
    max_samples_per_center: int = 25,
    image_size: int = 1024,
    split_type: str = "fold",
    fold_nr: int = 1,
    splits_root: Optional[str] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Build foreign amplitude spectrum pools across training clients.
    Excludes the out_center (Center 6) completely.
    """
    training_cids = [c for c in client_ids if str(c) != str(out_center)]
    print(f"\n[DFT-HT] Building amplitude pools across training clients: {training_cids} (Excluded out-center: {out_center})...")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 1. Pre-extract amplitude spectra for each training client
    center_amplitudes: Dict[str, List[torch.Tensor]] = {}
    for cid in training_cids:
        ds = PolypDataset(
            data_root=data_root,
            client_id=cid,
            mode="train",
            image_size=image_size,
            split_type=split_type,
            fold_nr=fold_nr,
            splits_root=splits_root,
        )
        n = min(len(ds), max_samples_per_center)
        indices = np.random.choice(len(ds), n, replace=False) if len(ds) > 0 else []
        client_amps = []
        for i in indices:
            img_path, _ = ds.samples[i]
            img = Image.open(img_path).convert("RGB")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            img_tensor = normalize(img_tensor)

            # Extract amplitude spectrum on CPU
            amp = extract_amp_spectrum(img_tensor).cpu()
            client_amps.append(amp)

        center_amplitudes[cid] = client_amps
        print(f"  Client {cid} ({ds.client_name}): extracted {len(client_amps)} amplitude spectra")

    # 2. For each training client, aggregate spectra from foreign training clients
    client_pools: Dict[str, List[torch.Tensor]] = {}
    for cid in training_cids:
        foreign_amps = []
        for foreign_cid in training_cids:
            if foreign_cid != cid:
                foreign_amps.extend(center_amplitudes[foreign_cid])
        client_pools[cid] = foreign_amps
        print(f"  Client {cid} total foreign amplitude pool: {len(foreign_amps)} spectra")

    # Out-center receives empty pool (it is only used for OOD testing)
    client_pools[str(out_center)] = []
    return client_pools


def get_client_dataloader(
    client_id: str,
    mode: str = "train",
    batch_size: int = 2,
    shuffle: Optional[bool] = None,
    num_workers: int = 2,
    data_root: str = DEFAULT_DATA_ROOT,
    image_size: int = 1024,
    target_amp_pool: Optional[List[torch.Tensor]] = None,
    dftht_prob: float = 0.3,
    dftht_L: float = 0.1,
    dftht_alpha: float = 0.05,
    metadata_dict: Optional[Dict[str, dict]] = None,
    split_type: str = "fold",
    fold_nr: int = 1,
    splits_root: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader with optional Fed-FGS DFT-HT augmentation, metadata, and fold splits."""
    if shuffle is None:
        shuffle = (mode == "train")

    dataset = PolypDataset(
        data_root=data_root,
        client_id=client_id,
        mode=mode,
        image_size=image_size,
        target_amp_pool=target_amp_pool if mode == "train" else None,
        dftht_prob=dftht_prob,
        dftht_L=dftht_L,
        dftht_alpha=dftht_alpha,
        metadata_dict=metadata_dict,
        split_type=split_type,
        fold_nr=fold_nr,
        splits_root=splits_root,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
