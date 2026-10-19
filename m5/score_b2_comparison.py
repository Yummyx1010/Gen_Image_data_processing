import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from uuid import uuid4

from evaluate import FIELDS
from score_predictions import calculate_metrics, read_json, require, sha256, write_csv


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("A", "F", "B", "B2")
SEEDS = (42, 123, 2026)
GENERATORS = ("glide", "Midjourney", "wukong")
NAMES = {"A": "m2_baseline_a", "F": "pretrained_frequency_only",
         "B": "baseline_b", "B2": "baseline_b_2"}
ORIGINAL_RUN = ROOT / "results/m5_test_20261008_124810_081d31"
ORIGINAL_SCORES = ROOT / "results/m5_scores_20261008_184008_9498f1/scores.json"


def checked_path(parent, name):
    path = (parent / name).resolve()
    require(path.is_relative_to(parent.resolve()), f"Invalid relative path: {name}")
    return path


def read_predictions(run_dir, models, trial_dir=None):
    run_dir = run_dir.resolve()
    require(run_dir.is_relative_to(ROOT / "results"), "Expected a formal prediction run")
    report = read_json(run_dir / "run_report.json")
    require(report["status"] == "complete" and report["category"] == "test", "Full test has not completed")
    require(report["sample_limit"] is None and report["samples_per_model"] == 8666, "Incomplete test")
    require(report["formal_inference_started"] and report["inference_performed"], "No full inference recorded")
    require(report["config"]["models"] == list(models) and report["config"]["seeds"] == list(SEEDS),
            "Unexpected model/seed configuration")
    protocol = report["prediction_protocol"]
    require(protocol["threshold"] == 0.5 and protocol["fake_label"] == 1 and protocol["real_label"] == 0,
            "Unexpected score direction or threshold")
    require(not protocol["shuffle"] and not protocol["balanced_sampling"], "Unexpected test sampling")
    expected_pairs = {(model, seed) for model in models for seed in SEEDS}
    items = report["models"]
    require(len(items) == len(expected_pairs) and {(x["model"], x["seed"]) for x in items} == expected_pairs,
            "Missing or duplicate checkpoints")
    validation = read_json(run_dir / "validation_report.json")
    require(validation["status"] == "complete" and validation["models_verified"] == len(expected_pairs)
            and validation["images_per_model"] == 8666, "Missing completed prediction verification")
    verified = {(x["model"], x["seed"]): x for x in validation["models"]}
    require(len(validation["models"]) == len(verified) and set(verified) == expected_pairs,
            "Unexpected verification records")
    test_csv = checked_path(ROOT, report["config"]["test_csv"])
    require(sha256(test_csv) == report["input_checks"]["test_csv_sha256"], "Test manifest changed")
    with test_csv.open(newline="", encoding="utf-8-sig") as stream:
        targets = list(csv.DictReader(stream))
    require(len(targets) == 8666, "Unexpected test size")
    require(Counter(x["generator"] for x in targets) ==
            {"Nature": 1166, "glide": 2500, "Midjourney": 2500, "wukong": 2500}, "Unexpected generators")
    identities = [(str(i), x["image_path"].replace("\\", "/"), x["generator"], x["label"])
                  for i, x in enumerate(targets)]
    require(len({x[1].casefold() for x in identities}) == 8666, "Duplicate test image")
    require(all(int(x[3]) == int(x[2] != "Nature") for x in identities), "Incorrect labels")
    metrics, evidence = [], []
    for item in items:
        path = checked_path(run_dir, item["predictions"]["csv"])
        require(sha256(path) == item["predictions"]["sha256"], f"Prediction checksum mismatch: {path.name}")
        record = verified[item["model"], item["seed"]]
        require(record["prediction_sha256"] == item["predictions"]["sha256"]
                and record["csv"] == path.name and record["rows"] == 8666,
                "Prediction does not match its verification record")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == FIELDS, f"Wrong CSV columns: {path.name}")
            rows = list(reader)
        require(len(rows) == item["predictions"]["rows"] == 8666, f"Incomplete predictions: {path.name}")
        for row, target in zip(rows, identities):
            require((row["row_index"], row["sample_id"], row["generator"], row["true_label"]) == target,
                    f"Test image identity mismatch: {path.name}")
            require(row["model"] == item["model"] and row["model_name"] == NAMES[item["model"]]
                    and row["seed"] == str(item["seed"]) and row["split"] == "test", "Prediction metadata mismatch")
            logit, probability = float(row["logit"]), float(row["pred_probability"])
            require(math.isfinite(logit) and math.isfinite(probability) and 0 <= probability <= 1,
                    "Invalid prediction value")
            sigmoid = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            require(math.isclose(probability, sigmoid, rel_tol=1e-6, abs_tol=1e-7), "Probability/logit mismatch")
            require(row["pred_label"] == str(int(probability >= 0.5)), "Classification threshold mismatch")
        if trial_dir is not None:
            with (trial_dir / path.name).open(newline="", encoding="utf-8-sig") as stream:
                trial = list(csv.DictReader(stream))
            require(len(trial) == 32, "Expected the 32-image B2 trial")
            require(all(a["sample_id"] == b["sample_id"] and
                        abs(float(a["logit"]) - float(b["logit"])) <= 1e-6
                        for a, b in zip(rows[:32], trial)), "Full predictions differ from the CPU trial")
        for generator in GENERATORS:
            subset = [x for x in rows if x["generator"] in ("Nature", generator)]
            measured = calculate_metrics([int(x["true_label"]) for x in subset],
                                         [float(x["pred_probability"]) for x in subset])
            metrics.append({"model": item["model"], "seed": item["seed"], "generator": generator, **measured})
        evidence.append({"model": item["model"], "seed": item["seed"], "csv": path.name,
                         "prediction_sha256": item["predictions"]["sha256"], "rows": len(rows)})
        print(f"Verified and scored {item['model']} seed {item['seed']}", flush=True)
    require(not list(run_dir.glob("*.part")), "Incomplete prediction files remain")
    actual_csvs = {p.name for p in run_dir.glob("*.csv") if not p.name.startswith("._")}
    require(actual_csvs == {x["csv"] for x in evidence}, "Unexpected prediction files")
    return metrics, evidence, report["input_checks"]["test_csv_sha256"]


def aggregate(rows):
    lookup = {(x["model"], x["seed"], x["generator"]): x for x in rows}
    expected = {(m, s, g) for m in MODELS for s in SEEDS for g in GENERATORS}
    require(len(rows) == len(lookup) == 36 and set(lookup) == expected, "Expected 36 distinct result groups")
    stability, generator_summary, summary, paired = [], [], [], []
    for model in MODELS:
        for seed in SEEDS:
            aucs = [lookup[model, seed, g]["auroc"] for g in GENERATORS]
            stability.append({"model": model, "seed": seed,
                              **{f"{g}_auroc": a for g, a in zip(GENERATORS, aucs)},
                              "macro_auroc": statistics.mean(aucs),
                              "cross_generator_sd": statistics.pstdev(aucs),
                              "worst_generator_auroc": min(aucs), "auroc_range": max(aucs) - min(aucs)})
        for generator in GENERATORS:
            group = [lookup[model, seed, generator] for seed in SEEDS]
            entry = {"model": model, "generator": generator, "n_seeds": 3}
            for metric in ("auroc", "accuracy", "precision", "recall"):
                values = [x[metric] for x in group if x[metric] is not None]
                entry[f"{metric}_mean"] = statistics.mean(values) if len(values) == 3 else None
                entry[f"{metric}_seed_sd"] = statistics.stdev(values) if len(values) == 3 else None
            generator_summary.append(entry)
        group = [x for x in stability if x["model"] == model]
        entry = {"model": model, "n_seeds": 3, "n_generators": 3}
        for metric in ("macro_auroc", "cross_generator_sd", "worst_generator_auroc", "auroc_range"):
            values = [x[metric] for x in group]
            entry[f"{metric}_mean"] = statistics.mean(values)
            entry[f"{metric}_seed_sd"] = statistics.stdev(values)
        entry.update({f"{g}_auroc_mean": statistics.mean(lookup[model, s, g]["auroc"] for s in SEEDS)
                      for g in GENERATORS})
        summary.append(entry)
    by_seed = {(x["model"], x["seed"]): x for x in stability}
    for reference in ("A", "B"):
        for seed in SEEDS:
            for metric in ("macro_auroc", "cross_generator_sd", "worst_generator_auroc"):
                a, b = by_seed[reference, seed][metric], by_seed["B2", seed][metric]
                paired.append({"reference": reference, "seed": seed, "metric": metric,
                               "reference_value": a, "B2": b, "B2_minus_reference": b - a,
                               "preferred_direction": "lower" if metric == "cross_generator_sd" else "higher"})
    return stability, generator_summary, summary, paired


def score(run_dir):
    status = read_json(ROOT / "logs/b2/M5_B2_STATUS.json")
    trial = checked_path(ROOT, status["diagnostic_run"])
    original, old_evidence, old_manifest = read_predictions(ORIGINAL_RUN, ("A", "F", "B"))
    new, new_evidence, new_manifest = read_predictions(run_dir, ("B2",), trial)
    require(old_manifest == new_manifest, "Original and B2 test sets differ")
    stability, generator_summary, summary, paired = aggregate(original + new)
    previous = read_json(ORIGINAL_SCORES)
    require(summary[:3] == previous["model_summary"], "Recalculated original scores differ from saved results")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    output = ROOT / "results/b2" / f"comparison_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    tables = {"per_generator_metrics": original + new, "per_seed_stability": stability,
              "generator_summary": generator_summary, "model_summary": summary, "paired_b2_comparison": paired}
    for name, values in tables.items():
        write_csv(output / f"{name}.csv", values)
    result = {"status": "complete", "created_at": datetime.now().astimezone().isoformat(),
              "b2_prediction_run": str(run_dir.resolve().relative_to(ROOT)),
              "original_prediction_run": str(ORIGINAL_RUN.relative_to(ROOT)),
              "original_scores_sha256": sha256(ORIGINAL_SCORES), "test_csv_sha256": new_manifest,
              "scoring_entrypoint_sha256": sha256(__file__), "protocol": previous["protocol"],
              "verification": {"prediction_files": 12, "prediction_rows": 103992,
                               "generator_seed_groups": 36, "original_scores_reproduced": True,
                               "b2_full_outputs_match_trial": True,
                               "original_validation_report_sha256": sha256(ORIGINAL_RUN / "validation_report.json"),
                               "b2_validation_report_sha256": sha256(run_dir / "validation_report.json")},
              "prediction_files": old_evidence + new_evidence, **tables}
    (output / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    status.update({"status": "full_test_scored", "formal_inference_started": True,
                   "full_test_metrics_computed": True, "b2_scoring_extension_prepared": True,
                   "full_run": str(run_dir.resolve().relative_to(ROOT)), "score_dir": str(output.relative_to(ROOT))})
    (ROOT / "logs/b2/M5_B2_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"score_dir": str(output), "model_summary": summary}, indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare original A/F/B results with frozen-frequency B2")
    parser.add_argument("--run-dir", type=Path, required=True)
    score(parser.parse_args().run_dir)
