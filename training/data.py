"""M1 data adapter and path/generator-disjoint protocol checks."""

import csv
import hashlib
from collections import Counter
from pathlib import PureWindowsPath

import torch
from torch.utils.data import DataLoader
from .config import resolve_path


def inspect_splits(config):
    report, seen_paths, seen_generators = {}, set(), set()
    for split in ("train", "validation", "test"):
        path = resolve_path(config.split_dir) / f"{split}.csv"
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"image_path", "label", "generator", "split"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: missing required CSV columns")
            rows = list(reader)
        paths, generators, counts = set(), set(), Counter()
        for row in rows:
            relative = (row["image_path"] or "").replace("\\", "/")
            win = PureWindowsPath(relative)
            if not relative or win.drive or relative.startswith("/") or ".." in win.parts:
                raise ValueError(f"{path}: image_path must be a safe relative path")
            canonical = win.as_posix().casefold()
            if canonical in paths or canonical in seen_paths:
                raise ValueError(f"Duplicate/overlapping image path: {relative}")
            if row["label"] not in {"0", "1"} or row["split"] != split or not row["generator"]:
                raise ValueError(f"{path}: invalid label, generator or split")
            paths.add(canonical)
            counts[row["label"]] += 1
            if row["label"] == "1":
                generators.add(row["generator"].casefold())
        if set(counts) != {"0", "1"}:
            raise ValueError(f"{path}: requires both real=0 and fake=1")
        if generators & seen_generators:
            raise ValueError("Fake generators overlap between splits")
        seen_paths.update(paths)
        seen_generators.update(generators)
        report[split] = {"samples": len(rows), "real": counts["0"], "fake": counts["1"],
                         "generators": dict(Counter(row["generator"] for row in rows)),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return report


def make_loader(config, split, generator=None):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split: {split}")
    if config.smoke:
        from .smoke import SyntheticDataset
        dataset = SyntheticDataset(split, config.seed)
    else:
        from dataset import GenImageDataset, transform
        root = resolve_path(config.data_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {root}")
        dataset = GenImageDataset(str(resolve_path(config.split_dir) / f"{split}.csv"), str(root), transform)
        # Only the in-memory DataFrame is normalized, never M1's source or CSVs.
        dataset.data["image_path"] = dataset.data["image_path"].str.replace("\\", "/", regex=False)
        for relative in dataset.data["image_path"]:
            if not (root / relative).is_file():
                raise FileNotFoundError(f"Missing {split} image: {root / relative}")
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=split == "train",
                      generator=generator, num_workers=0, drop_last=False)
