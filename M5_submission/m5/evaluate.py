import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile
import time
from unittest.mock import patch
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TMPDIR"] = str(PROJECT_ROOT / "logs" / "tmp")
tempfile.tempdir = os.environ["TMPDIR"]

ARCHITECTURES = {
    "A": ("m2_baseline_a", "m2_linear_v1", 512),
    "F": ("pretrained_frequency_only", "pretrained_frequency_only_v1", 256),
    "B": ("baseline_b", "m3_frequency_concat_v1", 768),
}
FIELDS = ["row_index", "sample_id", "generator", "split", "true_label",
          "logit", "pred_probability", "pred_label", "model", "model_name", "seed"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Empty relative path")
    value = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    if windows.drive or value.startswith("/") or ".." in windows.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    if value.endswith("/") or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Invalid relative path: {value}")
    return value


def project_path(value):
    path = (PROJECT_ROOT / relative_path(value)).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path escapes the project: {value}")
    return path


def validate_inputs(config, chosen_models, chosen_seeds):
    data_root = project_path(config["data_root"])
    test_csv = project_path(config["test_csv"])
    with test_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "label", "generator", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Test CSV is missing required columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        name = relative_path(row["image_path"])
        key = name.casefold()
        if key in seen:
            raise ValueError(f"Duplicate test image: {name}")
        seen.add(key)
        if row["split"] != "test" or row["label"] != str(int(row["generator"] != "Nature")):
            raise ValueError(f"Invalid test label or split: {name}")
        image = data_root / name
        if not image.resolve().is_relative_to(data_root) or not image.is_file():
            raise FileNotFoundError(f"Missing test image: {image}")
    counts = dict(Counter(row["generator"] for row in rows))
    if len(rows) != config["expected_images"]:
        raise ValueError(f"Unexpected test image count: {len(rows)}")
    if counts != config["expected_generators"]:
        raise ValueError(f"Unexpected test generators: {counts}")

    indexed = read_json(project_path(config["model_index"]))["models"]
    expected_pairs = {(model, seed) for model in config["models"] for seed in config["seeds"]}
    if len(indexed) != len(expected_pairs) or {(i["model"], i["seed"]) for i in indexed} != expected_pairs:
        raise ValueError("The model index must contain exactly A/F/B x three formal seeds")
    selected = [next(i for i in indexed if i["model"] == model and i["seed"] == seed)
                for model in chosen_models for seed in chosen_seeds]
    return rows, selected, {
        "test_images": len(rows), "generators": counts, "test_csv_sha256": sha256(test_csv),
    }


def load_model(item, config):
    import torch
    from torchvision.models import resnet18
    from train_final import load_regularized_checkpoint

    path = project_path(item["checkpoint"])
    if sha256(path) != item["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")

    def checkpoint_resnet18(*args, **kwargs):
        kwargs["weights"] = None
        return resnet18(*args, **kwargs)

    with ExitStack() as stack:
        for module in ("models.spatial_encoder", "models.frequency_encoder_pretrained"):
            stack.enter_context(patch(f"{module}.resnet18", checkpoint_resnet18))
        model, checkpoint = load_regularized_checkpoint(path, "cpu")
    model_name, architecture, feature_dim = ARCHITECTURES[item["model"]]
    if checkpoint["model_name"] != model_name or checkpoint["architecture"] != architecture:
        raise ValueError(f"Wrong architecture for {item['model']}")
    if checkpoint["config"]["seed"] != item["seed"]:
        raise ValueError(f"Checkpoint seed mismatch: {path}")
    if checkpoint["config"]["learning_rate"] != config["training_learning_rate"]:
        raise ValueError(f"Checkpoint learning rate mismatch: {path}")
    if checkpoint["selection_metric"] != "validation_auroc" or checkpoint["epoch"] != checkpoint["best_epoch"]:
        raise ValueError("Expected the best validation-AUROC checkpoint")
    if not isinstance(model.classifier, torch.nn.Linear) or model.classifier.weight.shape != (1, feature_dim):
        raise ValueError(f"Wrong linear head: {path}")
    model.eval().requires_grad_(False)
    metadata = {
        **item, "model_name": model_name, "architecture": architecture,
        "best_epoch": checkpoint["best_epoch"],
        "validation_auroc_at_selection": checkpoint["selected_checkpoint_validation_AUROC"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "strict_load_passed": True, "network_downloads": 0,
    }
    return model, metadata


def predict_to_csv(model, loader, path, item, device, threshold):
    import torch
    path = Path(path)
    temporary = path.with_suffix(".csv.part")
    count = 0
    started = time.perf_counter()
    model.eval()
    with temporary.open("x", newline="", encoding="utf-8") as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for batch in loader:
            logits = model(batch["image"].to(device=device, dtype=torch.float32))
            size = len(batch["path"])
            if logits.shape != (size, 1) or not torch.isfinite(logits).all().item():
                raise ValueError("Expected finite model logits with shape [batch,1]")
            values = logits.flatten().cpu().tolist()
            probabilities = torch.sigmoid(logits).flatten().cpu().tolist()
            labels = batch["label"].tolist()
            for index, (logit, probability, label) in enumerate(zip(values, probabilities, labels)):
                writer.writerow({
                    "row_index": count + index,
                    "sample_id": relative_path(batch["path"][index]),
                    "generator": batch["generator"][index], "split": "test",
                    "true_label": int(label), "logit": logit, "pred_probability": probability,
                    "pred_label": int(probability >= threshold), "model": item["model"],
                    "model_name": ARCHITECTURES[item["model"]][0], "seed": item["seed"],
                })
            count += size
            if count == size or count % (loader.batch_size * 20) == 0 or count == len(loader.dataset):
                elapsed = time.perf_counter() - started
                print(f"  {item['model']} seed {item['seed']}: {count}/{len(loader.dataset)} "
                      f"images, {elapsed:.1f}s", flush=True)
    if count != len(loader.dataset):
        raise ValueError(f"Incomplete prediction file: {count}/{len(loader.dataset)}")
    temporary.replace(path)
    seconds = time.perf_counter() - started
    return {"csv": path.name, "rows": count, "sha256": sha256(path),
            "seconds": seconds, "images_per_second": count / seconds}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="M5 model evaluation")
    parser.add_argument("--mode", choices=("check", "predict"), default="check")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("evaluation_config.json"))
    parser.add_argument("--models", nargs="+", choices=tuple(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 123, 2026))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.limit is not None and (args.limit < 1 or args.mode != "predict"):
        parser.error("--limit must be positive and used with --mode predict")
    for name in ("models", "seeds"):
        values = getattr(args, name)
        if values and len(values) != len(set(values)):
            parser.error(f"Duplicate --{name}")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = read_json(args.config)
    code_dir = project_path(config["code_dir"])
    sys.path.insert(0, str(code_dir))
    rows, selected, validation = validate_inputs(
        config, args.models or config["models"], args.seeds or config["seeds"])
    if args.limit is not None and args.limit > len(rows):
        raise ValueError("Diagnostic limit exceeds test size")

    import torch
    import torchvision
    from torch.utils.data import DataLoader, Subset
    from dataset import GenImageDataset, transform

    device = args.device or config["device"]
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unavailable inference device: {device}")
    if type(config["batch_size"]) is not int or config["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    if config["threshold"] != 0.5:
        raise ValueError("The prepared protocol fixes the decision threshold at 0.5")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = None
    if args.mode == "predict":
        dataset = GenImageDataset(str(project_path(config["test_csv"])),
                                  str(project_path(config["data_root"])), transform)
        dataset.data["image_path"] = dataset.data["image_path"].str.replace("\\", "/", regex=False)
        if args.limit is not None:
            dataset = Subset(dataset, range(args.limit))
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=0, drop_last=False)

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    category = "check" if args.mode == "check" else ("diagnostic" if args.limit is not None else "test")
    destination = config["results_dir"] if category == "test" else config["logs_dir"]
    run_dir = project_path(destination) / f"m5_{category}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "mode": args.mode, "category": category,
        "created_at": datetime.now().astimezone().isoformat(),
        "repository_commit": config["repository_commit"], "project_root": str(PROJECT_ROOT),
        "inference_performed": False, "formal_inference_started": False,
        "sample_limit": args.limit, "samples_per_model": args.limit or len(rows),
        "config": config, "config_sha256": sha256(args.config),
        "entrypoint_sha256": sha256(__file__), "input_checks": validation,
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "device": device,
                    "cpu_threads": torch.get_num_threads()},
        "prediction_protocol": {"real_label": 0, "fake_label": 1,
                                "score": "sigmoid(logit) = P(fake)", "threshold": config["threshold"],
                                "shuffle": False, "balanced_sampling": False, "num_workers": 0,
                                "preprocessing": "code/dataset.py transform"},
        "models": [],
    }
    save_json(run_dir / "run_report.json", report)
    try:
        print(f"Mode: {category}; test images: {len(rows)}; checkpoints: {len(selected)}", flush=True)
        for item in selected:
            print(f"Loading {item['model']} seed {item['seed']} (no download)...", flush=True)
            model, metadata = load_model(item, config)
            if args.mode == "predict":
                model.to(device)
                report["inference_performed"] = True
                report["formal_inference_started"] = category == "test"
                save_json(run_dir / "run_report.json", report)
                metadata["predictions"] = predict_to_csv(
                    model, loader, run_dir / f"{item['model']}_seed_{item['seed']}_predictions.csv",
                    item, device, config["threshold"])
            report["models"].append(metadata)
            save_json(run_dir / "run_report.json", report)
            del model
            gc.collect()
        report["status"] = "complete"
        report["completed_at"] = datetime.now().astimezone().isoformat()
        save_json(run_dir / "run_report.json", report)
        print(f"Complete. Record: {run_dir / 'run_report.json'}", flush=True)
        if args.mode == "check":
            print("Preparation check only: no images decoded, no predictions, no training.", flush=True)
        return 0
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save_json(run_dir / "run_report.json", report)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"M5 stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
