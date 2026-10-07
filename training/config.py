"""Dataclass + JSON configuration; paths resolve from the repository root."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass
class TrainConfig:
    baseline: str = "A"
    data_root: str = ""
    split_dir: str = "."
    output_dir: str = "results/training"
    run_name: Optional[str] = None
    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    frequency_factory: str = "models.frequency_encoder:FrequencyEncoder"
    frequency_kwargs: dict = field(default_factory=dict)
    frequency_input: Optional[str] = "normalized"
    smoke: bool = False

    def validate(self, require_data=True):
        if self.baseline not in {"A", "B"} or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("baseline must be A/B; device must be auto/cpu/cuda")
        for name in ("epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {name}")
        if self.run_name is not None and (not isinstance(self.run_name, str) or not self.run_name or self.run_name in {".", ".."} or Path(self.run_name).name != self.run_name):
            raise ValueError("run_name must be a single directory name")
        if self.num_workers != 0:
            raise ValueError("Keep num_workers=0: M1's unchanged Lambda transform is not spawn-picklable")
        if self.baseline == "B":
            if self.frequency_input not in {"normalized", "rgb01"}:
                raise ValueError("frequency_input must be normalized or rgb01")
            if ":" not in self.frequency_factory:
                raise ValueError("frequency_factory must use module:factory syntax")
            if not self.smoke and self.frequency_factory.startswith("training.smoke:"):
                raise ValueError("The smoke frequency stub must not be used for experiments")
        if require_data and not self.smoke and not self.data_root:
            raise ValueError("Provide --data-root containing Nature, ADM, BigGAN, etc.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        with resolve_path(path).open(encoding="utf-8-sig") as stream:
            return cls(**json.load(stream))
