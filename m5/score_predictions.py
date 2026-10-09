import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import uuid

sys.dont_write_bytecode = True
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("A", "F", "B")
SEEDS = (42, 123, 2026)
GENERATORS = ("glide", "Midjourney", "wukong")
METRICS = ("auroc", "accuracy", "precision", "recall")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_child(parent, name):
    path = (parent / name).resolve()
    require(path.is_relative_to(parent.resolve()), f"Unsafe relative path: {name}")
    return path


def calculate_metrics(labels, scores, threshold=0.5):
    require(len(labels) == len(scores) and len(labels) > 0, "Invalid metric inputs")
    require(set(labels) == {0, 1}, "Expected real=0 and fake=1, with both classes present")
    require(all(math.isfinite(x) and 0 <= x <= 1 for x in scores), "Invalid probability")
    predictions = [int(score >= threshold) for score in scores]
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    auroc = float(roc_auc_score(labels, scores))
    accuracy = float(accuracy_score(labels, predictions))
    precision_value = float(precision_score(labels, predictions, pos_label=1, zero_division=float("nan")))
    precision = precision_value if math.isfinite(precision_value) else None
    recall = float(recall_score(labels, predictions, pos_label=1))
    measured = {"auroc": auroc, "accuracy": accuracy, "precision": precision, "recall": recall}
    return {"n_images": len(labels), "n_real": len(labels) - sum(labels), "n_fake": sum(labels),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, **measured,
            "precision_defined": precision is not None}


def aggregate(per_generator):
    lookup = {(row["model"], row["seed"], row["generator"]): row for row in per_generator}
    require(len(lookup) == 27 and len(per_generator) == 27, "Expected 27 distinct model/seed/generator groups")
    stability = []
    for model in MODELS:
        for seed in SEEDS:
            aucs = [lookup[model, seed, generator]["auroc"] for generator in GENERATORS]
            minimum = min(aucs)
            stability.append({"model": model, "seed": seed,
                              **{f"{g}_auroc": a for g, a in zip(GENERATORS, aucs)},
                              "macro_auroc": statistics.mean(aucs),
                              "cross_generator_sd": statistics.pstdev(aucs),
                              "worst_generator_auroc": minimum,
                              "worst_generators": ";".join(g for g, a in zip(GENERATORS, aucs) if a == minimum),
                              "auroc_range": max(aucs) - minimum})
    generator_summary = []
    for model in MODELS:
        for generator in GENERATORS:
            group = [lookup[model, seed, generator] for seed in SEEDS]
            row = {"model": model, "generator": generator, "n_seeds": 3,
                   "n_images_per_seed": 3666, "n_real_per_seed": 1166, "n_fake_per_seed": 2500}
            for metric in METRICS:
                values = [x[metric] for x in group if x[metric] is not None]
                row[f"{metric}_n_defined"] = len(values)
                row[f"{metric}_mean"] = statistics.mean(values) if len(values) == 3 else None
                row[f"{metric}_seed_sd"] = statistics.stdev(values) if len(values) == 3 else None
            generator_summary.append(row)
    model_summary = []
    for model in MODELS:
        group = [x for x in stability if x["model"] == model]
        row = {"model": model, "n_seeds": 3, "n_generators": 3}
        for metric in ("macro_auroc", "cross_generator_sd", "worst_generator_auroc", "auroc_range"):
            values = [x[metric] for x in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_seed_sd"] = statistics.stdev(values)
        for generator in GENERATORS:
            row[f"{generator}_auroc_mean"] = statistics.mean(lookup[model, seed, generator]["auroc"] for seed in SEEDS)
        model_summary.append(row)
    paired = []
    stable_lookup = {(x["model"], x["seed"]): x for x in stability}
    for seed in SEEDS:
        for generator in GENERATORS:
            a, b = lookup["A", seed, generator]["auroc"], lookup["B", seed, generator]["auroc"]
            paired.append({"seed": seed, "scope": generator, "metric": "auroc", "A": a, "B": b,
                           "B_minus_A": b - a, "preferred_direction": "higher"})
        for metric in ("macro_auroc", "cross_generator_sd", "worst_generator_auroc"):
            a, b = stable_lookup["A", seed][metric], stable_lookup["B", seed][metric]
            paired.append({"seed": seed, "scope": "three_generators", "metric": metric,
                           "A": a, "B": b, "B_minus_A": b - a,
                           "preferred_direction": "lower" if metric == "cross_generator_sd" else "higher"})
    return stability, generator_summary, model_summary, paired


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(run_dir):
    started = time.perf_counter()
    run_dir = run_dir.resolve()
    require(run_dir.is_relative_to(ROOT / "results"), "Expected predictions under this project's results/")
    report_path, validation_path = run_dir / "run_report.json", run_dir / "validation_report.json"
    report, validation = read_json(report_path), read_json(validation_path)
    require(report["status"] == validation["status"] == "complete", "Predictions must be complete and verified")
    require(report["category"] == "test" and report["sample_limit"] is None, "A diagnostic subset cannot be scored as the full test")
    require(report["config"]["models"] == list(MODELS) and report["config"]["seeds"] == list(SEEDS), "Unexpected model/seed selection")
    require(report["samples_per_model"] == 8666, "Unexpected test size")
    require(report["prediction_protocol"]["threshold"] == 0.5 and report["prediction_protocol"]["fake_label"] == 1,
            "Unexpected score direction or threshold")
    require(report["prediction_protocol"]["shuffle"] is False and report["prediction_protocol"]["balanced_sampling"] is False,
            "Unexpected test sampling")
    expected_pairs = {(model, seed) for model in MODELS for seed in SEEDS}
    require(len(report["models"]) == 9, "Expected nine checkpoints")
    require({(x["model"], x["seed"]) for x in report["models"]} == expected_pairs, "Missing or duplicate checkpoint")
    expected_csv = safe_child(ROOT, report["config"]["test_csv"])
    manifest_hash = sha256(expected_csv)
    require(manifest_hash == report["input_checks"]["test_csv_sha256"], "Test manifest changed")
    with expected_csv.open(newline="", encoding="utf-8-sig") as stream:
        targets = list(csv.DictReader(stream))
    require(len(targets) == 8666, "Incomplete test manifest")
    target_identities = [(str(i), row["image_path"].replace("\\", "/"), row["generator"], row["label"])
                         for i, row in enumerate(targets)]
    require(len({x[1].casefold() for x in target_identities}) == 8666, "Duplicate test image")
    require(Counter(x[2] for x in target_identities) == {"Nature": 1166, "glide": 2500, "Midjourney": 2500, "wukong": 2500},
            "Unexpected generator counts")
    require(all(int(x[3]) == (0 if x[2] == "Nature" else 1) for x in target_identities), "Unexpected real/fake labels")
    inputs = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in (report_path, validation_path)]
    inputs.append({"path": str(expected_csv.relative_to(ROOT)), "sha256": manifest_hash})
    per_generator, pooled = [], []
    for item in report["models"]:
        key = item["model"], item["seed"]
        path = safe_child(run_dir, item["predictions"]["csv"])
        digest = sha256(path)
        require(digest == item["predictions"]["sha256"], f"Prediction CSV changed: {path.name}")
        inputs.append({"path": str(path.relative_to(ROOT)), "sha256": digest})
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        identities = [(row["row_index"], row["sample_id"], row["generator"], row["true_label"]) for row in rows]
        require(identities == target_identities, f"Test images or labels differ: {path.name}")
        require(all(row["split"] == "test" and row["model"] == key[0] and row["seed"] == str(key[1]) for row in rows),
                "Prediction metadata mismatch")
        for generator in GENERATORS:
            subset = [row for row in rows if row["generator"] in ("Nature", generator)]
            result = calculate_metrics([int(x["true_label"]) for x in subset], [float(x["pred_probability"]) for x in subset])
            per_generator.append({"model": key[0], "seed": key[1], "generator": generator, **result})
        pooled.append({"model": key[0], "seed": key[1], "scope": "all_generators_pooled_secondary",
                       **calculate_metrics([int(x["true_label"]) for x in rows], [float(x["pred_probability"]) for x in rows])})
        print(f"Scored {key[0]} seed {key[1]}: three generators, shared 1166 real images", flush=True)
    stability, generator_summary, model_summary, paired = aggregate(per_generator)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "results" / f"m5_scores_{stamp}_{uuid.uuid4().hex[:6]}"
    output.mkdir(exist_ok=False)
    result = {
        "status": "complete", "created_at": datetime.now().astimezone().isoformat(),
        "prediction_run": str(run_dir.relative_to(ROOT)), "score_dir": str(output.relative_to(ROOT)),
        "repository_commit": report["repository_commit"], "scoring_entrypoint_sha256": sha256(__file__),
        "runtime": {"python": sys.version, "executable": sys.executable, "sklearn": sklearn_version},
        "inputs": inputs,
        "protocol": {
            "positive_class": "AI-generated = 1", "negative_class": "Nature real = 0",
            "auroc_score": "saved pred_probability = P(AI-generated); ties receive half credit",
            "classification_threshold": 0.5, "threshold_was_tuned": False,
            "each_generator_cohort": {"real": 1166, "fake": 2500, "total": 3666},
            "same_real_images_reused_for_all_generators": True,
            "macro_auroc": "unweighted mean of the three generator-specific AUROCs, separately for each seed",
            "cross_generator_sd": "population standard deviation (ddof=0) across the three fixed generators, separately for each seed",
            "worst_generator_auroc": "minimum of the three generator-specific AUROCs, separately for each seed; then averaged over seeds",
            "seed_summary": "unweighted mean and sample standard deviation (ddof=1) across seeds 42,123,2026; not an ensemble",
            "undefined_precision": "null/blank, not zero; three-seed mean unavailable if any seed is undefined",
            "pooled_metrics": "secondary only; each of the 8666 unique images included once per checkpoint",
            "interpretation": "cross-generator SD must be interpreted together with macro and worst-generator AUROC; seed SD is not a confidence interval",
        },
        "verification": {"prediction_files": 9, "prediction_rows": 77994, "generator_seed_groups": 27,
                         "method": "prediction hashes and test image identities checked; metrics calculated with sklearn"},
        "per_generator": per_generator, "pooled_secondary": pooled, "per_seed_stability": stability,
        "generator_summary": generator_summary, "model_summary": model_summary, "paired_b_vs_a": paired,
        "training_performed": False, "inference_performed": False,
        "plots_created": False, "error_case_analysis_performed": False,
    }
    for name, rows in (("per_generator_metrics", per_generator), ("pooled_metrics_secondary", pooled),
                       ("per_seed_stability", stability), ("generator_summary", generator_summary),
                       ("model_summary", model_summary), ("paired_b_vs_a", paired)):
        write_csv(output / f"{name}.csv", rows)
    result["elapsed_seconds"] = time.perf_counter() - started
    (output / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (ROOT / "logs/M5_METRICS_STATUS.json").write_text(json.dumps({"status": "scores_calculated",
          "score_dir": str(output.relative_to(ROOT)), "scores_json": str((output / "scores.json").relative_to(ROOT)),
          "updated_at": result["created_at"], "plots_created": False, "error_case_analysis_performed": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"score_dir": str(output), "elapsed_seconds": result["elapsed_seconds"],
                      "model_summary": model_summary}, ensure_ascii=False, indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate M5 test metrics")
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        score(arguments.run_dir)
    except (ValueError, KeyError, OSError) as error:
        print(f"Scoring failed: {error}", file=sys.stderr)
        raise SystemExit(1)
