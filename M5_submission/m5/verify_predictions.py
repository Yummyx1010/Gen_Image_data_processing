"""Verify a completed full M5 prediction run, without training or inference."""
import argparse
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {"A": "m2_baseline_a", "F": "pretrained_frequency_only", "B": "baseline_b"}
FIELDS = ["row_index", "sample_id", "generator", "split", "true_label", "logit",
          "pred_probability", "pred_label", "model", "model_name", "seed"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def child_path(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root), f"Path escapes its directory: {name}")
    return path


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def verify(run_dir):
    run_dir = run_dir.resolve()
    require(run_dir.is_relative_to(ROOT / "results"), "Expected a run under this project's results/")
    report = read_json(run_dir / "run_report.json")
    require(report["status"] == "complete", "The prediction run has not completed successfully")
    require(report["category"] == "test" and report["sample_limit"] is None,
            "Diagnostic subsets must not be used as full test results")
    require(report["formal_inference_started"] and report["inference_performed"], "No full inference recorded")
    config = report["config"]
    require(config["models"] == ["A", "F", "B"] and config["seeds"] == [42, 123, 2026],
            "Expected all three models and all three formal seeds")
    pairs = {(item["model"], item["seed"]) for item in report["models"]}
    expected_pairs = {(model, seed) for model in config["models"] for seed in config["seeds"]}
    require(len(report["models"]) == 9 and pairs == expected_pairs, "Missing or duplicate model/seed results")
    require(config["threshold"] == report["prediction_protocol"]["threshold"] == 0.5,
            "Unexpected classification threshold")
    require(report["prediction_protocol"]["shuffle"] is False and
            report["prediction_protocol"]["balanced_sampling"] is False, "Unexpected test sampling")
    code_dir = child_path(ROOT, config["code_dir"])
    test_csv = child_path(ROOT, config["test_csv"])
    require(sha256(test_csv) == report["input_checks"]["test_csv_sha256"], "Test CSV changed during inference")
    _, expected = read_csv(test_csv)
    require(len(expected) == report["samples_per_model"] == config["expected_images"] == 8666,
            "Unexpected full test size")
    require(len({row["image_path"].replace("\\", "/").casefold() for row in expected}) == len(expected),
            "Duplicate test image paths")
    counts = dict(Counter(row["generator"] for row in expected))
    require(counts == config["expected_generators"], "Unexpected test generator counts")

    # Recheck team-source bytes and checkpoint files after prediction has finished.
    manifest = read_json(child_path(ROOT, config["code_manifest"]))
    for item in manifest["files"]:
        content = child_path(code_dir, item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        require(len(content) == item["bytes"] and blob == item["git_blob_sha"],
                f"Original source changed: {item['path']}")

    trial = read_json(ROOT / "logs/M5_STEP2_TRIAL_REPORT.json")
    trial_dir = child_path(ROOT, trial["diagnostic_run"])
    summaries = []
    for item in report["models"]:
        prediction = item["predictions"]
        path = child_path(run_dir, prediction["csv"])
        require(sha256(path) == prediction["sha256"], f"Prediction checksum mismatch: {path.name}")
        fields, rows = read_csv(path)
        require(fields == FIELDS and len(rows) == prediction["rows"] == len(expected),
                f"Incomplete/invalid prediction table: {path.name}")
        for index, (row, target) in enumerate(zip(rows, expected)):
            identity = (row["row_index"], row["sample_id"], row["true_label"], row["generator"], row["split"])
            target_identity = (str(index), target["image_path"].replace("\\", "/"),
                               target["label"], target["generator"], "test")
            require(identity == target_identity, f"Row identity mismatch: {path.name}:{index + 2}")
            require(row["model"] == item["model"] and row["model_name"] == MODEL_NAMES[item["model"]]
                    and row["seed"] == str(item["seed"]), f"Model identity mismatch: {path.name}")
            logit, probability = float(row["logit"]), float(row["pred_probability"])
            require(math.isfinite(logit) and math.isfinite(probability) and 0 <= probability <= 1,
                    f"Non-finite/invalid prediction: {path.name}:{index + 2}")
            sigmoid = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            require(math.isclose(probability, sigmoid, rel_tol=1e-6, abs_tol=1e-7),
                    f"Probability/logit mismatch: {path.name}:{index + 2}")
            require(row["pred_label"] == str(int(probability >= 0.5)),
                    f"Predicted label mismatch: {path.name}:{index + 2}")
        checkpoint = child_path(ROOT, item["checkpoint"])
        require(checkpoint.stat().st_size == item["bytes"] and sha256(checkpoint) == item["sha256"],
                f"Checkpoint changed during inference: {checkpoint}")
        _, trial_rows = read_csv(trial_dir / path.name)
        require(len(trial_rows) == 64, f"Unexpected diagnostic reference size: {path.name}")
        max_difference = max(abs(float(a["logit"]) - float(b["logit"]))
                             for a, b in zip(rows[:64], trial_rows))
        require(max_difference <= 1e-6, f"Full run differs from its earlier CPU trial: {path.name}")
        summaries.append({"model": item["model"], "seed": item["seed"], "rows": len(rows),
                          "csv": path.name, "prediction_sha256": prediction["sha256"],
                          "checkpoint_sha256": item["sha256"], "seconds": prediction["seconds"],
                          "trial_max_absolute_logit_difference": max_difference})
        print(f"Verified {item['model']} seed {item['seed']}: {len(rows)} rows", flush=True)
    require(not [path for path in run_dir.iterdir() if path.name.endswith(".part")], "Incomplete output files remain")
    csv_files = {path.name for path in run_dir.glob("*.csv") if not path.name.startswith("._")}
    require(csv_files == {item["csv"] for item in summaries}, "Unexpected/missing prediction CSVs")
    verified = {
        "status": "complete", "verified_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir.relative_to(ROOT)), "repository_commit": report["repository_commit"],
        "models_verified": 9, "images_per_model": len(expected), "total_prediction_rows": 9 * len(expected),
        "generators": counts, "runtime": report["runtime"],
        "original_team_files_verified_unchanged": len(manifest["files"]),
        "model_checkpoint_files_verified_unchanged": 9,
        "checks": ["all model/seed pairs present", "exact CSV identities and order",
                   "finite logits and probabilities", "sigmoid and fixed threshold consistent",
                   "prediction and checkpoint SHA256 matched", "full outputs match earlier CPU trial",
                   "no incomplete .part files"],
        "prediction_seconds_total": sum(item["seconds"] for item in summaries),
        "models": summaries,
        "test_performance_metrics_computed": False,
        "training_performed": False,
        "next_step": "Calculate metrics and produce tables/plots from these verified predictions",
    }
    output = run_dir / "validation_report.json"
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(verified, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(f"Complete: {verified['total_prediction_rows']} rows. Record: {output}", flush=True)
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        verify(arguments.run_dir)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
