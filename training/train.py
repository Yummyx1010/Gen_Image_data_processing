"""Unified A/B runner. Run from the repository root: python -m src.training.train."""

import argparse
from contextlib import nullcontext
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

from models.baseline_a_adapter import FrozenBaselineA
from models.baseline_b import BaselineB
from .config import ROOT, TrainConfig, resolve_path
from .data import inspect_splits, make_loader


PREDICTION_FIELDS = ["sample_id", "generator", "split", "true_label",
                     "pred_probability", "model_name", "seed"]


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def device_for(config):
    name = config.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use cpu or auto")
    return torch.device(name)


def build_model(config):
    config.validate(require_data=False)
    context = nullcontext()
    if config.smoke:
        from .smoke import offline_spatial_initialization
        context = offline_spatial_initialization()
    with context:
        if config.baseline == "A":
            return FrozenBaselineA()
        module, factory = config.frequency_factory.split(":", 1)
        try:
            frequency_factory = getattr(importlib.import_module(module), factory)
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"Frequency encoder is unavailable: {config.frequency_factory}. Supply an nn.Module "
                "including agreed FFT/log-spectrum preprocessing and returning [B,256]."
            ) from error
        frequency = frequency_factory(**config.frequency_kwargs)
        return BaselineB(frequency, config.frequency_input)


def run_epoch(model, loader, device, optimizer=None, grad_clip=1.0, prediction_context=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    loss_sum, correct, count = 0.0, 0, 0
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device=device, dtype=torch.float32).reshape(-1, 1)
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("Labels must be real=0, fake=1")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            if logits.shape != labels.shape:
                raise ValueError(f"Expected logits {labels.shape}; got {logits.shape}")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                frequency = getattr(model, "frequency_encoder", None)
                if frequency is not None and not any(p.grad is not None for p in frequency.parameters()):
                    raise RuntimeError("No gradients reached frequency encoder parameters")
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         grad_clip, error_if_nonfinite=True)
                optimizer.step()
        size = labels.numel()
        loss_sum += loss.item() * size
        correct += ((logits.detach() >= 0) == labels.bool()).sum().item()
        count += size
        if prediction_context is not None:
            probabilities = torch.sigmoid(logits.detach()).reshape(-1).cpu().tolist()
            for index, probability in enumerate(probabilities):
                rows.append({"sample_id": batch["path"][index].replace("\\", "/"),
                             "generator": batch["generator"][index],
                             "split": prediction_context["split"],
                             "true_label": int(labels[index].item()),
                             "pred_probability": probability,
                             "model_name": prediction_context["model_name"],
                             "seed": prediction_context["seed"]})
    if not count:
        raise ValueError("Empty DataLoader")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}, rows


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_rng(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path):
    checkpoint = torch.load(resolve_path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected an M4 unified-training checkpoint, not M2's debug state_dict")
    return checkpoint


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def source_hashes():
    files = [ROOT / "dataset.py", *sorted((ROOT / "models").rglob("*.py")),
             *sorted((ROOT / "training").rglob("*.py"))]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def train(config, resume=None):
    config.validate()
    seed_everything(config.seed)
    if config.smoke:
        torch.set_num_threads(1)
    device = device_for(config)
    audit = {"smoke": True} if config.smoke else inspect_splits(config)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = make_loader(config, "train", generator)
    validation_loader = make_loader(config, "validation")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_loss, start_epoch = [], float("inf"), 1
    if resume:
        checkpoint = load_checkpoint(resume)
        differences = [key for key, value in config.to_dict().items()
                       if key != "epochs" and checkpoint["config"][key] != value]
        if differences or checkpoint["data_audit"] != audit or checkpoint["source_sha256"] != source_hashes():
            raise ValueError(f"Resume requires unchanged settings, CSVs and model/training code: {differences}")
        run_dir = resolve_path(resume).parent
        if Path(resume).name != "last.pt" or not (run_dir / "best.pt").exists():
            raise ValueError("Resume from last.pt and retain best.pt in the same directory")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history, best_loss = checkpoint["history"], checkpoint["best_validation_loss"]
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch > config.epochs:
            raise ValueError("Total --epochs must exceed completed epochs")
        restore_rng(checkpoint["rng"], generator)
    else:
        prefix = "SMOKE" if config.smoke else "Baseline"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = resolve_path(config.output_dir) / f"{prefix}_{config.baseline}_seed{config.seed}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", config.to_dict())
    save_json(run_dir / "data_audit.json", audit)
    save_json(run_dir / "environment.json", {
        "python": sys.version, "torch": str(torch.__version__), "numpy": str(np.__version__),
        "device": str(device), "smoke": config.smoke, "source_sha256": source_hashes(),
        "spatial_weights": "random offline test only" if config.smoke else "M2 ResNet18_Weights.DEFAULT",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })
    for epoch in range(start_epoch, config.epochs + 1):
        training, _ = run_epoch(model, train_loader, device, optimizer, config.grad_clip)
        validation, _ = run_epoch(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": training["loss"],
                        "train_accuracy": training["accuracy"], "validation_loss": validation["loss"],
                        "validation_accuracy": validation["accuracy"]})
        improved = validation["loss"] < best_loss
        best_loss = min(best_loss, validation["loss"])
        checkpoint = {"format_version": 1, "epoch": epoch, "config": config.to_dict(),
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "history": history, "best_validation_loss": best_loss, "rng": capture_rng(generator),
                      "data_audit": audit, "source_sha256": source_hashes()}
        save_checkpoint(run_dir / "last.pt", checkpoint)
        if improved:
            save_checkpoint(run_dir / "best.pt", checkpoint)
        save_csv(run_dir / "history.csv", history)
        print(f"epoch {epoch}/{config.epochs}: train loss={training['loss']:.4f}, "
              f"validation loss={validation['loss']:.4f}, accuracy={validation['accuracy']:.4f}")
    print(f"Saved run: {run_dir}")
    return run_dir


def export_predictions(checkpoint_path, split="test", device_override=None, data_root=None):
    if split not in {"validation", "test"}:
        raise ValueError("Export split must be validation or test")
    checkpoint = load_checkpoint(checkpoint_path)
    config = TrainConfig(**checkpoint["config"])
    # Evaluation may move devices/images; architecture, normalization and seed are retained.
    if device_override is not None:
        config.device = device_override
    if data_root is not None:
        config.data_root = data_root
    config.validate()
    seed_everything(config.seed)
    if config.smoke:
        torch.set_num_threads(1)
    if not config.smoke and inspect_splits(config) != checkpoint["data_audit"]:
        raise ValueError("CSV protocol differs from the saved checkpoint")
    device = device_for(config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, rows = run_epoch(model, make_loader(config, split), device, prediction_context={
        "split": split, "model_name": f"Baseline{config.baseline}", "seed": config.seed})
    output = resolve_path(checkpoint_path).parent / f"{Path(checkpoint_path).stem}_{split}_predictions.csv"
    save_csv(output, rows, PREDICTION_FIELDS)
    save_json(output.with_suffix(".json"), {**metrics, "smoke": config.smoke, "split": split,
              "epoch": checkpoint["epoch"], "threshold": 0.5, "positive_class": "fake",
              "effective_config": config.to_dict(), "checkpoint": str(resolve_path(checkpoint_path))})
    print(f"Predictions: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--config", help="JSON configuration, relative to repository root")
    mode.add_argument("--resume", help="Continue a run from last.pt")
    mode.add_argument("--predict", help="Export a checkpoint's seven-column M5 predictions")
    parser.add_argument("--baseline", choices=["A", "B"])
    parser.add_argument("--data-root")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--inspect-data", action="store_true", help="Audit only CSV metadata")
    parser.add_argument("--smoke", action="store_true", help="Offline synthetic check, never a formal experiment")
    args = parser.parse_args()
    if args.predict or args.resume:
        if args.config or args.baseline or args.seeds or args.smoke or (args.predict and args.epochs):
            parser.error("Checkpoint modes use saved model settings")
        if args.predict:
            checkpoint = load_checkpoint(args.predict)
            if checkpoint["config"]["smoke"]:
                print("SMOKE ONLY: exported probabilities are not experiment results.")
            export_predictions(args.predict, args.split, args.device, args.data_root)
            return
        if args.device or args.data_root:
            parser.error("Resume keeps the original device/data root; change only --epochs")
        config = TrainConfig(**load_checkpoint(args.resume)["config"])
    else:
        config = TrainConfig.load(args.config) if args.config else TrainConfig()
        if args.smoke:
            config = replace(config, smoke=True, baseline=args.baseline or "B", device="cpu",
                             epochs=2, batch_size=4, output_dir="results/smoke",
                             frequency_factory="training.smoke:FrequencyStub", frequency_input="rgb01")
        for name in ("baseline", "data_root", "device"):
            value = getattr(args, name)
            if value is not None:
                setattr(config, name, value)
    if args.inspect_data:
        print(json.dumps(inspect_splits(config), indent=2))
        return
    if args.epochs is not None:
        config.epochs = args.epochs
    if config.smoke:
        print("SMOKE ONLY: random ResNet-18, RGB frequency stub, synthetic images. NOT experiment results.")
    seeds = args.seeds if args.seeds is not None else [config.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds must be distinct")
    for seed in seeds:
        train(replace(config, seed=seed), args.resume)


if __name__ == "__main__":
    main()
