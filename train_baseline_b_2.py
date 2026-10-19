"""Train/evaluate Baseline B_2 without changing the existing A/B runners.

Example (PowerShell):
  & $py train_baseline_b_2.py --seed 123 --data-root $dataRoot
  & $py train_baseline_b_2.py --predict 'results/baseline_b_2/lr_9e-4/<run>/best.pt' --data-root $dataRoot
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from models.baseline_b_2 import BaselineB2
from train_final import (
    classification_metrics, improved_selection, lr_tag, make_balanced_train_loader,
)
from training.config import ROOT, TrainConfig, resolve_path
from training.data import inspect_splits, make_loader
from training.train import (
    PREDICTION_FIELDS, device_for, save_checkpoint, save_csv, save_json,
    seed_everything, source_hashes,
)


MODEL_NAME = "baseline_b_2"
ARCHITECTURE = "m3_frozen_frequency_concat_v1"
DEFAULT_PROTOCOL = "configs/baseline_b_2_lr9e-4.json"


def load_protocol(path):
    """Require the old B protocol's hyperparameters to remain exactly intact."""
    with resolve_path(path).open(encoding="utf-8-sig") as stream:
        protocol = json.load(stream)
    with (ROOT / "configs/result_compare_lr9e-4_b.json").open(encoding="utf-8-sig") as stream:
        reference = json.load(stream)
    expected = {
        **reference,
        "phase": "baseline_b_2_lr9e-4",
        "architecture": ARCHITECTURE,
        "model_description": (
            "Baseline B_2: frozen spatial ResNet-18 plus M3's fully frozen frequency "
            "ResNet-18 and 512-to-256 projection; concatenate 512+256 features "
            "and train only Linear(768, 1). The projection is seeded but not pretrained."
        ),
        "model_source": [
            "models/spatial_encoder.py",
            "models/frequency_encoder_frozen.py",
            "models/baseline_b_2.py",
        ],
        "freeze_frequency_backbone": True,
        "freeze_frequency_encoder": True,
        "output_dir": "results/baseline_b_2",
    }
    if protocol != expected:
        differences = sorted(key for key in protocol.keys() | expected.keys()
                             if protocol.get(key) != expected.get(key))
        raise ValueError(f"B_2 protocol differs from the controlled B settings: {differences}")
    return protocol


def run_epoch_frozen(model, loader, device, optimizer=None, grad_clip=1.0,
                     prediction_context=None):
    """Same epoch/loss logic as the existing runner, without a frequency-gradient requirement."""
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
                if any(parameter.grad is not None for parameter in model.spatial_encoder.parameters()):
                    raise RuntimeError("Spatial encoder unexpectedly received gradients")
                if any(parameter.grad is not None for parameter in model.frequency_encoder.parameters()):
                    raise RuntimeError("Frozen frequency encoder unexpectedly received gradients")
                nn.utils.clip_grad_norm_(model.classifier.parameters(), grad_clip,
                                         error_if_nonfinite=True)
                optimizer.step()
        size = labels.numel()
        loss_sum += loss.item() * size
        correct += ((logits.detach() >= 0) == labels.bool()).sum().item()
        count += size
        if prediction_context is not None:
            probabilities = torch.sigmoid(logits.detach()).reshape(-1).cpu().tolist()
            for index, probability in enumerate(probabilities):
                rows.append({
                    "sample_id": batch["path"][index].replace("\\", "/"),
                    "generator": batch["generator"][index],
                    "split": prediction_context["split"],
                    "true_label": int(labels[index].item()),
                    "pred_probability": probability,
                    "model_name": MODEL_NAME,
                    "seed": prediction_context["seed"],
                })
    if not count:
        raise ValueError("Empty DataLoader")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}, rows


def make_config(seed, data_root, protocol):
    config = TrainConfig(
        baseline="B", data_root=data_root, seed=seed,
        epochs=protocol["max_epochs"], batch_size=protocol["batch_size"],
        learning_rate=protocol["learning_rate"], weight_decay=protocol["weight_decay"],
        grad_clip=protocol["grad_clip"], device=protocol["device"],
        frequency_factory="models.frequency_encoder_frozen:FrozenFrequencyEncoder",
        frequency_kwargs={"feature_dim": 256}, frequency_input="normalized",
    )
    config.validate()
    return config


def run_training(seed, data_root, protocol):
    config = make_config(seed, data_root, protocol)
    seed_everything(seed)
    audit = inspect_splits(config)
    device = device_for(config)
    train_loader = make_balanced_train_loader(config)
    validation_loader = make_loader(config, "validation")
    model = BaselineB2().to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable) != 2 or any(parameter is not expected for parameter, expected in
                                  zip(trainable, model.classifier.parameters())):
        raise RuntimeError("Only the B_2 fusion classifier may be trainable")
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (resolve_path(protocol["output_dir"]) / f"lr_{lr_tag(config.learning_rate)}" /
               f"{MODEL_NAME}_seed{seed}_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=False)
    config.output_dir, config.run_name = str(run_dir.parent), run_dir.name
    hashes = source_hashes()
    hashes["train_final.py"] = hashlib.sha256((ROOT / "train_final.py").read_bytes()).hexdigest()
    hashes["train_baseline_b_2.py"] = hashlib.sha256((ROOT / "train_baseline_b_2.py").read_bytes()).hexdigest()
    hardware = {
        "device": str(device), "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_gb": round(
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 3), 2
        ) if device.type == "cuda" else None,
    }
    save_json(run_dir / "config.json", config.to_dict())
    save_json(run_dir / "protocol.json", protocol)
    save_json(run_dir / "data_audit.json", audit)
    save_json(run_dir / "environment.json", {
        **hardware, "source_sha256": hashes,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    })

    history = []
    metric = protocol["selection_metric"]
    best_selection = float("-inf") if metric == "validation_auroc" else float("inf")
    best_loss, best_auroc, best_epoch, stale = float("inf"), float("-inf"), 0, 0
    best_loss_metrics = best_auroc_metrics = selected_validation = None
    selected_validation_loss = None
    start = time.perf_counter()
    log_path = run_dir / f"training_log_{MODEL_NAME}_seed{seed}.csv"
    print(f"Training {MODEL_NAME} with {ARCHITECTURE} on {device}", flush=True)
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        training, train_rows = run_epoch_frozen(
            model, train_loader, device, optimizer, config.grad_clip,
            prediction_context={"split": "train_sampled", "seed": seed},
        )
        validation, validation_rows = run_epoch_frozen(
            model, validation_loader, device,
            prediction_context={"split": "validation", "seed": seed},
        )
        train_metrics = classification_metrics(train_rows)
        validation_metrics = classification_metrics(validation_rows)
        history.append({
            "epoch": epoch, "train_loss": training["loss"],
            "validation_loss": validation["loss"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "epoch_seconds": time.perf_counter() - epoch_start,
        })
        if validation["loss"] < best_loss:
            best_loss, best_loss_metrics = validation["loss"], validation_metrics
        if validation_metrics["auroc"] > best_auroc:
            best_auroc, best_auroc_metrics = validation_metrics["auroc"], validation_metrics
        current = validation_metrics["auroc"] if metric == "validation_auroc" else validation["loss"]
        improved = improved_selection(current, best_selection, metric,
                                      protocol["early_stopping_min_delta"])
        if improved:
            best_selection = current
            best_epoch, selected_validation, selected_validation_loss, stale = (
                epoch, validation_metrics, validation["loss"], 0)
        elif protocol["early_stopping_enabled"] and epoch > protocol["early_stopping_warmup_epochs"]:
            stale += 1
        else:
            stale = 0
        checkpoint = {
            "format_version": 2, "architecture": ARCHITECTURE,
            "model_name": MODEL_NAME, "epoch": epoch, "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history, "best_validation_loss": best_loss,
            "best_validation_AUROC": best_auroc,
            "selection_metric": metric,
            "selected_checkpoint_validation_loss": selected_validation_loss,
            "selected_checkpoint_validation_AUROC": selected_validation["auroc"],
            "best_epoch": best_epoch, "data_audit": audit,
            "source_sha256": hashes, "protocol": protocol,
        }
        last_path = save_checkpoint(run_dir / "last.pt", checkpoint, allow_fallback=True)
        if improved:
            best_path = save_checkpoint(run_dir / "best.pt", checkpoint, allow_fallback=True)
        save_csv(log_path, history)
        stop_reason = (
            "early_stopping" if protocol["early_stopping_enabled"] and
            epoch > protocol["early_stopping_warmup_epochs"] and
            stale >= protocol["early_stopping_patience"] else
            "maximum_epochs" if epoch == config.epochs else "in_progress"
        )
        save_json(run_dir / "run_summary.json", {
            "model": MODEL_NAME, "architecture": ARCHITECTURE, "seed": seed,
            "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
            "batch_size": config.batch_size, "maximum_epochs": config.epochs,
            "epochs_completed": epoch, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "best_validation_AUROC": best_auroc,
            "best_validation_metrics_at_best_loss": best_loss_metrics,
            "best_validation_metrics_at_best_auroc": best_auroc_metrics,
            "selection_metric": metric,
            "selected_checkpoint_validation_loss": selected_validation_loss,
            "selected_checkpoint_validation_AUROC": selected_validation["auroc"],
            "training_seconds": time.perf_counter() - start,
            "hardware": hardware, "checkpoint_path": str(best_path.resolve()),
            "last_checkpoint_path": str(last_path.resolve()),
            "stop_reason": stop_reason,
            "sampling": "balanced training only; official validation distribution",
            "frozen_encoders": ["spatial", "frequency_backbone", "frequency_projection"],
        })
        print(
            f"epoch {epoch}/{config.epochs} | "
            f"train loss={training['loss']:.4f} AUROC={train_metrics['auroc']:.4f} | "
            f"val loss={validation['loss']:.4f} AUROC={validation_metrics['auroc']:.4f} | "
            f"best epoch={best_epoch}", flush=True,
        )
        if stop_reason == "early_stopping":
            print(f"Early stopping: {metric} did not improve for {stale} epochs.", flush=True)
            break
    print(f"Saved run: {run_dir}", flush=True)
    return run_dir


def load_b2_checkpoint(path, device="cpu"):
    checkpoint = torch.load(resolve_path(path), map_location=device, weights_only=True)
    if (checkpoint.get("format_version"), checkpoint.get("architecture"),
            checkpoint.get("model_name")) != (2, ARCHITECTURE, MODEL_NAME):
        raise ValueError("This is not a Baseline B_2 checkpoint")
    model = BaselineB2().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def export_predictions(checkpoint_path, data_root, split="validation"):
    if split not in {"validation", "test"}:
        raise ValueError("Prediction split must be validation or test")
    stored = torch.load(resolve_path(checkpoint_path), map_location="cpu", weights_only=True)
    if (stored.get("architecture"), stored.get("model_name")) != (ARCHITECTURE, MODEL_NAME):
        raise ValueError("This is not a Baseline B_2 checkpoint")
    config = TrainConfig(**stored["config"])
    config.data_root = data_root
    config.validate()
    seed_everything(config.seed)
    audit = inspect_splits(config)
    if audit != stored["data_audit"]:
        raise ValueError("CSV split audit differs from the checkpoint")
    device = device_for(config)
    model, checkpoint = load_b2_checkpoint(checkpoint_path, device)
    _, rows = run_epoch_frozen(
        model, make_loader(config, split), device,
        prediction_context={"split": split, "seed": config.seed},
    )
    metrics = classification_metrics(rows)
    output = resolve_path(checkpoint_path).parent / f"{Path(checkpoint_path).stem}_{split}_predictions.csv"
    save_csv(output, rows, PREDICTION_FIELDS)
    save_json(output.with_suffix(".json"), {
        **metrics, "split": split, "epoch": checkpoint["epoch"],
        "threshold": 0.5, "positive_class": "fake",
        "checkpoint": str(resolve_path(checkpoint_path)),
        "architecture": ARCHITECTURE, "model_name": MODEL_NAME,
    })
    print(f"Predictions: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--predict", help="B_2 checkpoint to evaluate")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.predict:
        export_predictions(args.predict, args.data_root, args.split)
        return
    if args.seed is None:
        parser.error("--seed is required for training")
    protocol = load_protocol(args.protocol)
    if args.inspect_data:
        config = make_config(args.seed, args.data_root, protocol)
        print(json.dumps(inspect_splits(config), indent=2))
        return
    run_training(args.seed, args.data_root, protocol)


if __name__ == "__main__":
    main()
