"""One training protocol for spatial-only, frequency-only and fused models.

Train and validation images are used here; test images remain reserved for
evaluation after the model and hyperparameters have been selected.
"""
import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.baseline_a_adapter import FrozenBaselineA
from models.baseline_b import BaselineB
from models.frequency_encoder import FrequencyEncoder
from models.frequency_only import FrequencyOnlyModel
from models.pretrained_baseline_b import LegacyPretrainedBaselineB, PretrainedBaselineB
from training.config import ROOT, TrainConfig, resolve_path
from training.data import inspect_splits, make_loader
from training.train import (
    device_for, run_epoch, save_checkpoint, save_csv, save_json,
    seed_everything, source_hashes,
)

MODEL_NAMES = ("m2_baseline_a", "pretrained_frequency_only", "baseline_b", "pretrained_baseline_b")
ARCHITECTURES = {
    "m2_linear_v1", "pretrained_frequency_only_v1", "frequency_cnn_concat_v1",
    "pretrained_fusion_v1", "pretrained_fusion_linear_v2",
}


def load_protocol(path):
    with resolve_path(path).open(encoding="utf-8-sig") as stream:
        protocol = json.load(stream)
    if protocol["optimizer"] != "Adam" or protocol["loss"] != "BCEWithLogitsLoss":
        raise ValueError("Expected Adam and BCEWithLogitsLoss")
    if protocol["sampler"] != "WeightedRandomSampler":
        raise ValueError("Balanced sampling must be training-only")
    if protocol["selection_metric"] not in {"validation_loss", "validation_auroc"}:
        raise ValueError("Checkpoint selection must use validation loss or AUROC")
    if protocol["architecture"] not in ARCHITECTURES:
        raise ValueError("Unsupported training architecture")
    if protocol["learning_rate"] not in protocol["learning_rate_candidates"]:
        raise ValueError("Learning rate must be a declared candidate")
    for key in ("batch_size", "max_epochs"):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if protocol["architecture"] == "pretrained_fusion_v1":
        if type(protocol["hidden_dim"]) is not int or protocol["hidden_dim"] < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(protocol["dropout"], (int, float)) or not 0 <= protocol["dropout"] < 1:
            raise ValueError("Dropout must be in [0,1)")
    elif protocol["hidden_dim"] is not None or protocol["dropout"] is not None:
        raise ValueError("Linear models do not use hidden_dim or dropout")
    if protocol["architecture"].startswith("pretrained_"):
        if type(protocol["freeze_frequency_backbone"]) is not bool:
            raise ValueError("freeze_frequency_backbone must be a boolean")
    elif protocol["architecture"] == "frequency_cnn_concat_v1" and "freeze_frequency_backbone" in protocol:
        raise ValueError("The lightweight frequency CNN has no pretrained backbone to freeze")
    if type(protocol["early_stopping_enabled"]) is not bool:
        raise ValueError("early_stopping_enabled must be a boolean")
    if protocol["early_stopping_enabled"]:
        for key in ("early_stopping_patience", "early_stopping_warmup_epochs"):
            if type(protocol[key]) is not int or protocol[key] < 1:
                raise ValueError(f"{key} must be a positive integer when early stopping is enabled")
    for key in ("learning_rate", "weight_decay", "grad_clip", "early_stopping_min_delta"):
        value = protocol[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if protocol["learning_rate"] == 0 or protocol["grad_clip"] == 0:
        raise ValueError("Learning rate and gradient clipping must be positive")
    backbone_lr = protocol.get("frequency_backbone_learning_rate")
    if backbone_lr is not None:
        if protocol["architecture"] not in {"pretrained_fusion_v1", "pretrained_fusion_linear_v2"} or protocol["freeze_frequency_backbone"]:
            raise ValueError("Separate backbone LR requires a trainable pretrained fusion backbone")
        if not isinstance(backbone_lr, (int, float)) or not math.isfinite(backbone_lr):
            raise ValueError("Invalid frequency backbone learning rate")
        if not 0 < backbone_lr < protocol["learning_rate"]:
            raise ValueError("Frequency backbone LR must be positive and below the head LR")
    return protocol


def balanced_weights(labels):
    labels = [int(value) for value in labels]
    counts = Counter(labels)
    if set(counts) != {0, 1}:
        raise ValueError("Training labels must contain real=0 and fake=1")
    return torch.tensor([1.0 / counts[value] for value in labels], dtype=torch.double)


def make_balanced_train_loader(config):
    dataset = make_loader(config, "train").dataset
    sampler = WeightedRandomSampler(
        balanced_weights(dataset.data["label"]),
        num_samples=len(dataset), replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    return DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                      num_workers=0, drop_last=False)


def classification_metrics(rows):
    labels = [int(row["true_label"]) for row in rows]
    scores = [float(row["pred_probability"]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Both classes are needed for AUROC")
    predicted = [int(score >= 0.5) for score in scores]
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def lr_tag(value):
    scientific = f"{Decimal(str(value)).normalize():e}"
    return scientific.replace(".", "p").replace("e-0", "e-").replace("e+0", "e+")


def make_optimizer(model, protocol):
    head_lr = protocol["learning_rate"]
    backbone_lr = protocol.get("frequency_backbone_learning_rate")
    if backbone_lr is None:
        return torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=head_lr, weight_decay=protocol["weight_decay"],
        )
    backbone = [parameter for parameter in model.frequency_encoder.backbone.parameters()
                if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone}
    head = [parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids]
    if not backbone or not head:
        raise ValueError("Both backbone and projection/classifier parameters must be trainable")
    return torch.optim.Adam(
        [{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}],
        weight_decay=protocol["weight_decay"],
    )


def improved_selection(current, best, metric, min_delta):
    if metric == "validation_auroc":
        return current > best + min_delta
    if metric == "validation_loss":
        return current < best - min_delta
    raise ValueError(f"Unsupported checkpoint selection metric: {metric}")


def load_regularized_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    architecture = checkpoint.get("architecture")
    if checkpoint.get("format_version") != 2 or architecture not in ARCHITECTURES:
        raise ValueError("Unsupported training checkpoint")
    protocol = checkpoint["protocol"]
    if architecture == "m2_linear_v1":
        if checkpoint["model_name"] != "m2_baseline_a":
            raise ValueError("Checkpoint model and architecture do not match")
        model = FrozenBaselineA()
    elif architecture == "pretrained_frequency_only_v1":
        if checkpoint["model_name"] != "pretrained_frequency_only":
            raise ValueError("Checkpoint model and architecture do not match")
        model = FrequencyOnlyModel(
            feature_dim=256, freeze_backbone=protocol["freeze_frequency_backbone"]
        )
    elif architecture == "pretrained_fusion_v1":
        if checkpoint["model_name"] != "pretrained_baseline_b":
            raise ValueError("Checkpoint model and architecture do not match")
        model = LegacyPretrainedBaselineB(
            protocol["hidden_dim"], protocol["dropout"],
            protocol["freeze_frequency_backbone"], seed=checkpoint["config"]["seed"],
        )
    elif architecture == "pretrained_fusion_linear_v2":
        if checkpoint["model_name"] != "pretrained_baseline_b":
            raise ValueError("Checkpoint model and architecture do not match")
        model = PretrainedBaselineB(
            freeze_backbone=protocol["freeze_frequency_backbone"],
            seed=checkpoint["config"]["seed"],
        )
    elif architecture == "frequency_cnn_concat_v1":
        if checkpoint["model_name"] != "baseline_b":
            raise ValueError("Checkpoint model and architecture do not match")
        model = BaselineB(FrequencyEncoder(feature_dim=256), "normalized")
    else:
        raise ValueError("Unsupported training checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def train_final(model_name, seed, data_root, protocol):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Model must be one of {MODEL_NAMES}")
    architecture_for_model = {
        "m2_baseline_a": "m2_linear_v1",
        "pretrained_frequency_only": "pretrained_frequency_only_v1",
        "baseline_b": "frequency_cnn_concat_v1",
        "pretrained_baseline_b": {"pretrained_fusion_v1", "pretrained_fusion_linear_v2"},
    }
    expected_architecture = architecture_for_model[model_name]
    if isinstance(expected_architecture, str):
        expected_architecture = {expected_architecture}
    if protocol["architecture"] not in expected_architecture:
        raise ValueError("Model and protocol architecture must match")
    uses_pretrained_frequency = model_name in {"pretrained_frequency_only", "pretrained_baseline_b"}
    frequency_factory = {
        "m2_baseline_a": "not_applicable",
        "baseline_b": "models.frequency_encoder:FrequencyEncoder",
        "pretrained_frequency_only": "models.frequency_encoder_pretrained:PretrainedFrequencyEncoder",
        "pretrained_baseline_b": "models.frequency_encoder_pretrained:PretrainedFrequencyEncoder",
    }[model_name]
    frequency_kwargs = (
        {} if model_name == "m2_baseline_a" else
        {"feature_dim": 256} if model_name == "baseline_b" else
        {"feature_dim": 256, "freeze_backbone": protocol["freeze_frequency_backbone"]}
    )
    config = TrainConfig(
        baseline="A" if model_name == "m2_baseline_a" else "B",
        data_root=data_root, seed=seed, epochs=protocol["max_epochs"],
        batch_size=protocol["batch_size"], learning_rate=protocol["learning_rate"],
        weight_decay=protocol["weight_decay"], grad_clip=protocol["grad_clip"],
        device=protocol["device"],
        frequency_factory=frequency_factory,
        frequency_kwargs=frequency_kwargs,
        frequency_input=None if model_name == "m2_baseline_a" else "normalized",
    )
    config.validate()
    seed_everything(seed)
    audit = inspect_splits(config)
    device = device_for(config)
    train_loader = make_balanced_train_loader(config)
    validation_loader = make_loader(config, "validation")
    if model_name == "m2_baseline_a":
        model = FrozenBaselineA().to(device)
    elif model_name == "pretrained_frequency_only":
        model = FrequencyOnlyModel(
            feature_dim=256, freeze_backbone=protocol["freeze_frequency_backbone"]
        ).to(device)
    elif model_name == "baseline_b":
        model = BaselineB(FrequencyEncoder(feature_dim=256), "normalized").to(device)
    elif model_name == "pretrained_baseline_b":
        if protocol["architecture"] == "pretrained_fusion_v1":
            model = LegacyPretrainedBaselineB(
                protocol["hidden_dim"], protocol["dropout"],
                protocol["freeze_frequency_backbone"], seed=seed,
            ).to(device)
        else:
            model = PretrainedBaselineB(
                freeze_backbone=protocol["freeze_frequency_backbone"], seed=seed,
            ).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    optimizer = make_optimizer(model, protocol)
    print(f"Training {model_name} with {protocol['architecture']} on {device}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_label = (f"{model_name}_linear_v2"
                 if protocol["architecture"] == "pretrained_fusion_linear_v2"
                 else f"{model_name}_cnn_concat_v1"
                 if protocol["architecture"] == "frequency_cnn_concat_v1"
                 else model_name)
    run_dir = (resolve_path(protocol["output_dir"]) /
               f"lr_{lr_tag(config.learning_rate)}" /
               f"{run_label}_seed{seed}_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=False)
    config.output_dir, config.run_name = str(run_dir.parent), run_dir.name
    log_path = run_dir / f"training_log_{model_name}_seed{seed}.csv"
    hashes = source_hashes()
    hashes["train_final.py"] = hashlib.sha256((ROOT / "train_final.py").read_bytes()).hexdigest()
    if uses_pretrained_frequency:
        sources = ["models/frequency_encoder_pretrained.py"]
        sources.append("models/frequency_only.py" if model_name == "pretrained_frequency_only"
                       else "models/pretrained_baseline_b.py")
        for source in sources:
            hashes[source] = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    elif model_name == "baseline_b":
        for source in ("models/frequency_encoder.py", "models/baseline_b.py"):
            hashes[source] = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
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
    save_json(run_dir / "environment.json", {**hardware, "source_sha256": hashes})

    history = []
    selection_metric = protocol["selection_metric"]
    best_selection = float("-inf") if selection_metric == "validation_auroc" else float("inf")
    best_loss, best_auroc = float("inf"), float("-inf")
    best_loss_metrics = best_auroc_metrics = selected_validation = None
    selected_validation_loss = None
    best_epoch, stale = 0, 0
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        training, train_rows = run_epoch(
            model, train_loader, device, optimizer, config.grad_clip,
            prediction_context={"split": "train_sampled", "model_name": model_name, "seed": seed},
        )
        validation, validation_rows = run_epoch(
            model, validation_loader, device,
            prediction_context={"split": "validation", "model_name": model_name, "seed": seed},
        )
        train_metrics = classification_metrics(train_rows)
        validation_metrics = classification_metrics(validation_rows)
        row = {
            "epoch": epoch, "train_loss": training["loss"],
            "validation_loss": validation["loss"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss, best_loss_metrics = validation["loss"], validation_metrics
        if validation_metrics["auroc"] > best_auroc:
            best_auroc, best_auroc_metrics = validation_metrics["auroc"], validation_metrics
        current_selection = (validation_metrics["auroc"] if selection_metric == "validation_auroc"
                             else validation["loss"])
        improved = improved_selection(
            current_selection, best_selection, selection_metric,
            protocol["early_stopping_min_delta"],
        )
        if improved:
            best_selection = current_selection
            best_epoch, selected_validation, selected_validation_loss, stale = (
                epoch, validation_metrics, validation["loss"], 0)
        elif protocol["early_stopping_enabled"] and epoch > protocol["early_stopping_warmup_epochs"]:
            stale += 1
        else:
            stale = 0
        checkpoint = {
            "format_version": 2, "architecture": protocol["architecture"],
            "model_name": model_name, "epoch": epoch, "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history, "best_validation_loss": best_loss,
            "best_validation_AUROC": best_auroc,
            "selection_metric": selection_metric,
            "selected_checkpoint_validation_loss": selected_validation_loss,
            "selected_checkpoint_validation_AUROC": selected_validation["auroc"],
            "best_epoch": best_epoch, "data_audit": audit,
            "source_sha256": hashes, "protocol": protocol,
        }
        last_checkpoint_path = save_checkpoint(
            run_dir / "last.pt", checkpoint, allow_fallback=True,
        )
        if improved:
            best_checkpoint_path = save_checkpoint(
                run_dir / "best.pt", checkpoint, allow_fallback=True,
            )
        save_csv(log_path, history)
        stop_reason = (
            "early_stopping" if protocol["early_stopping_enabled"] and epoch > protocol["early_stopping_warmup_epochs"] and stale >= protocol["early_stopping_patience"]
            else "maximum_epochs" if epoch == config.epochs else "in_progress"
        )
        save_json(run_dir / "run_summary.json", {
            "model": model_name, "architecture": protocol["architecture"],
            "seed": seed, "learning_rate": config.learning_rate,
            "frequency_backbone_learning_rate": protocol.get("frequency_backbone_learning_rate"),
            "weight_decay": config.weight_decay, "dropout": protocol["dropout"],
            "hidden_dim": protocol["hidden_dim"],
            "batch_size": config.batch_size, "maximum_epochs": config.epochs, "early_stopping_enabled": protocol["early_stopping_enabled"], "early_stopping_warmup_epochs": protocol["early_stopping_warmup_epochs"], "early_stopping_patience": protocol["early_stopping_patience"],
            "epochs_completed": epoch, "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "best_validation_AUROC": best_auroc,
            "best_validation_metrics_at_best_loss": best_loss_metrics,
            "best_validation_metrics_at_best_auroc": best_auroc_metrics,
            "selection_metric": selection_metric,
            "selected_checkpoint_validation_loss": selected_validation_loss,
            "selected_checkpoint_validation_AUROC": selected_validation["auroc"],
            "training_seconds": time.perf_counter() - start,
            "hardware": hardware,
            "checkpoint_path": str(best_checkpoint_path.resolve()),
            "last_checkpoint_path": str(last_checkpoint_path.resolve()),
            "stop_reason": stop_reason,
            "sampling": "balanced training only; official validation distribution",
        })
        print(
            f"epoch {epoch}/{config.epochs} | "
            f"train loss={training['loss']:.4f} accuracy={train_metrics['accuracy']:.4f} "
            f"precision={train_metrics['precision']:.4f} recall={train_metrics['recall']:.4f} "
            f"AUROC={train_metrics['auroc']:.4f} | "
            f"val loss={validation['loss']:.4f} accuracy={validation_metrics['accuracy']:.4f} "
            f"precision={validation_metrics['precision']:.4f} "
            f"recall={validation_metrics['recall']:.4f} "
            f"AUROC={validation_metrics['auroc']:.4f} | best epoch={best_epoch}",
            flush=True,
        )
        if stop_reason == "early_stopping":
            print(f"Early stopping: {selection_metric} did not improve for {stale} epochs.", flush=True)
            break
    print(f"Saved run: {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--inspect-data", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.inspect_data:
        config = TrainConfig(data_root=args.data_root)
        print(json.dumps(inspect_splits(config), indent=2))
        return
    train_final(args.model, args.seed, args.data_root, protocol)


if __name__ == "__main__":
    main()
