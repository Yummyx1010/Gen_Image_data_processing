import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/m5_test_20261008_124810_081d31"
SCORES = ROOT / "results/m5_scores_20261008_184008_9498f1"
MODELS = ("A", "F", "B")
SEEDS = (42, 123, 2026)
SOURCES = ("glide", "Midjourney", "wukong", "Nature")
PATTERNS = ("both_correct", "A_correct_B_wrong", "A_wrong_B_correct", "both_wrong")
TITLES = {
    "both_correct": "A and B correct",
    "A_correct_B_wrong": "A correct, B wrong",
    "A_wrong_B_correct": "A wrong, B correct",
    "both_wrong": "A and B wrong",
    "mixed_across_seeds": "Pattern differs across seeds",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pattern(a_correct, b_correct):
    return {
        (True, True): "both_correct",
        (True, False): "A_correct_B_wrong",
        (False, True): "A_wrong_B_correct",
        (False, False): "both_wrong",
    }[(a_correct, b_correct)]


def confusion(rows):
    counts = Counter((r["true_label"], r["pred_label"]) for r in rows)
    result = {"tn": counts[0, 0], "fp": counts[0, 1],
              "fn": counts[1, 0], "tp": counts[1, 1]}
    result["n_images"] = len(rows)
    result["errors"] = result["fp"] + result["fn"]
    return result


def main():
    scores = json.loads((SCORES / "scores.json").read_text())
    run = json.loads((RUN / "run_report.json").read_text())
    assert scores["status"] == run["status"] == "complete"
    assert run["samples_per_model"] == 8666 and len(run["models"]) == 9
    threshold = scores["protocol"]["classification_threshold"]
    assert threshold == 0.5 and not scores["protocol"]["threshold_was_tuned"]

    protected = {}
    for item in scores["inputs"]:
        path = ROOT / item["path"]
        assert sha256(path) == item["sha256"], f"Source changed: {path}"
        protected[path] = item["sha256"]
    protected[SCORES / "scores.json"] = sha256(SCORES / "scores.json")
    expected_prediction_paths = {RUN / f"{m}_seed_{s}_predictions.csv"
                                 for m in MODELS for s in SEEDS}
    assert expected_prediction_paths <= set(protected)

    manifest = read_csv(ROOT / "code/test.csv")
    identities = [(r["image_path"].replace("\\", "/"), r["generator"],
                   r["split"], int(r["label"])) for r in manifest]
    assert len(identities) == len({r[0] for r in identities}) == 8666
    assert Counter(r[1] for r in identities) == {
        "glide": 2500, "Midjourney": 2500, "wukong": 2500, "Nature": 1166}
    assert all(split == "test" and label == (source != "Nature")
               for _, source, split, label in identities)

    predictions = {}
    for model in MODELS:
        for seed in SEEDS:
            rows = read_csv(RUN / f"{model}_seed_{seed}_predictions.csv")
            actual_ids = [(r["sample_id"], r["generator"], r["split"], int(r["true_label"]))
                          for r in rows]
            assert actual_ids == identities, f"Identity/order mismatch: {model}, {seed}"
            for row in rows:
                row["true_label"] = int(row["true_label"])
                row["pred_label"] = int(row["pred_label"])
                row["pred_probability"] = float(row["pred_probability"])
                assert row["pred_label"] == int(row["pred_probability"] >= threshold)
            predictions[model, seed] = rows

    per_seed = []
    all_patterns = {}
    for seed in SEEDS:
        groups = {source: {p: {"n": 0, "F_correct": 0, "F_wrong": 0}
                           for p in PATTERNS} for source in SOURCES}
        seed_patterns = []
        for i, (_, source, _, label) in enumerate(identities):
            correct = {m: predictions[m, seed][i]["pred_label"] == label for m in MODELS}
            category = pattern(correct["A"], correct["B"])
            seed_patterns.append(category)
            count = groups[source][category]
            count["n"] += 1
            count["F_correct" if correct["F"] else "F_wrong"] += 1
        all_patterns[seed] = seed_patterns
        totals = {p: sum(groups[source][p]["n"] for source in SOURCES) for p in PATTERNS}
        model_counts = {m: {source: confusion([r for r in predictions[m, seed]
                                             if r["generator"] == source])
                            for source in SOURCES} for m in MODELS}
        errors = {m: sum(v["errors"] for v in model_counts[m].values()) for m in MODELS}
        per_seed.append({"seed": seed, "pattern_counts": totals, "model_errors": errors,
                         "patterns_by_source": groups, "confusion_by_model_and_source": model_counts})

    consensus = {}
    consensus_counts = {p: {s: 0 for s in SOURCES} for p in (*PATTERNS, "mixed_across_seeds")}
    for i, (sample_id, source, _, _) in enumerate(identities):
        seen = {all_patterns[seed][i] for seed in SEEDS}
        category = seen.pop() if len(seen) == 1 else "mixed_across_seeds"
        consensus[sample_id] = category
        consensus_counts[category][source] += 1
    by_id = {r[0]: i for i, r in enumerate(identities)}
    coverage = Counter({source: 0 for source in SOURCES})
    cases = []
    for category in ("A_correct_B_wrong", "A_wrong_B_correct", "both_wrong"):
        category_sources = set()
        for _ in range(2):
            available = [source for source in SOURCES if source not in category_sources
                         and any(p == category and identities[by_id[sid]][1] == source
                                 for sid, p in consensus.items())]
            source = min(available, key=lambda s: (coverage[s], SOURCES.index(s)))
            sample_id = min(sid for sid, p in consensus.items()
                            if p == category and identities[by_id[sid]][1] == source)
            i = by_id[sample_id]
            image_path = ROOT / "data" / sample_id
            assert image_path.is_file(), f"Candidate image missing: {image_path}"
            case = {"case_number": len(cases) + 1, "sample_id": sample_id,
                    "generator": source, "true_label": identities[i][3],
                    "true_class": "Real image" if source == "Nature" else "AI-generated image",
                    "pattern_all_three_seeds": category,
                    "pattern_title": TITLES[category],
                    "image_path": str(image_path), "image_sha256": sha256(image_path),
                    "visually_reviewed": False,
                    "correct_seeds": {}, "predictions": {}}
            protected[image_path] = case["image_sha256"]
            for model in MODELS:
                case["correct_seeds"][model] = sum(predictions[model, seed][i]["pred_label"] == identities[i][3]
                                                   for seed in SEEDS)
                case["predictions"][model] = {str(seed): {
                    "probability_ai": predictions[model, seed][i]["pred_probability"],
                    "pred_label": predictions[model, seed][i]["pred_label"],
                    "correct": predictions[model, seed][i]["pred_label"] == identities[i][3],
                } for seed in SEEDS}
            cases.append(case)
            category_sources.add(source)
            coverage[source] += 1
    assert all(coverage[s] > 0 for s in SOURCES)

    output = ROOT / "results" / f"m5_error_step1_original_{uuid.uuid4().hex[:8]}"
    output.mkdir()
    result = {
        "status": "step1_counts_complete_step2_candidates_prepared",
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "original_full_test_all_nine_checkpoints",
        "prediction_run": str(RUN.relative_to(ROOT)),
        "score_dir": str(SCORES.relative_to(ROOT)),
        "models": MODELS, "seeds": SEEDS, "unique_test_images": 8666,
        "prediction_rows": 77994, "classification_threshold": threshold,
        "positive_class": "AI-generated = 1", "negative_class": "Nature real = 0",
        "source_counting": "Each Nature image counted once per seed in these source error tables.",
        "patterns": TITLES, "per_seed": per_seed, "consensus_by_source": consensus_counts,
        "consensus_definition": "A/B correctness pattern identical in all three training seeds.",
        "verification": {"same_sample_identity_and_order_all_nine": True,
                         "source_hashes_match_original_scoring": True,
                         "candidate_images_present": True},
        "visual_error_analysis_complete": False,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": h} for p, h in protected.items()],
        "builder": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))},
    }
    save_json(output / "error_counts.json", result)
    save_json(output / "candidate_cases.json", {
        "status": "candidates_prepared_not_visually_reviewed",
        "selection_rule": "Two cases per error pattern; identical A/B pattern in all three seeds; prefer least-covered source, then fixed source order glide/Midjourney/wukong/Nature, then filename order.",
        "purpose": "Qualitative illustration only, not a replacement test set or a prevalence estimate.",
        "cases": cases,
    })

    lines = ["# Error counts on the original full test set", "",
             "The original A, F and B models each use seeds 42, 123 and 2026: nine checkpoints in total. Every checkpoint was evaluated on all 8,666 original test images.", "",
             "An image is classified as AI when P(AI) >= 0.5. These are fixed-threshold classification counts, not AUROC. Fewer errors do not necessarily imply higher AUROC.", "",
             "| Seed | A and B correct | A correct, B wrong | A wrong, B correct | A and B wrong | A errors | F errors | B errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in per_seed:
        counts, errors = item["pattern_counts"], item["model_errors"]
        values = [item["seed"], *(counts[p] for p in PATTERNS), *(errors[m] for m in MODELS)]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "The first four counts in each row sum to 8,666. All three seeds are reported separately.", "",
              "## Errors by image source", "",
              "| Seed | Source | Images | A errors | F errors | B errors |",
              "|---|---|---:|---:|---:|---:|"]
    for item in per_seed:
        for source in SOURCES:
            counts = item["confusion_by_model_and_source"]
            values = [item["seed"], source, counts["A"][source]["n_images"],
                      *(counts[m][source]["errors"] for m in MODELS)]
            lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "Nature contains real images; the other three sources contain AI-generated images. Each real image is counted once here. Generator-specific AUROC evaluations reuse the same real images for each generator.", "",
              "## A/B patterns shared by all three seeds", "",
              "| Pattern | glide | Midjourney | wukong | Nature | Total |",
              "|---|---:|---:|---:|---:|---:|"]
    for category, counts in consensus_counts.items():
        values = [TITLES[category], *(counts[s] for s in SOURCES), sum(counts.values())]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "## Selected cases", "",
              "Six cases were selected for the gallery. Sample IDs are in CANDIDATE_CASES.md; the selection method and analysis are in ERROR_ANALYSIS.md.", ""]
    (output / "ERROR_COUNTS.md").write_text("\n".join(lines), encoding="utf-8")

    candidate_lines = ["# Candidate cases before visual inspection", "",
                       "Each image has the same A/B correctness pattern across all three seeds. Counts show correct classifications out of three seeds. Per-seed probabilities and labels are in candidate_cases.json.", "",
                       "| Case | Pattern | True class | Image | A correct | F correct | B correct |",
                       "|---|---|---|---|---:|---:|---:|"]
    for c in cases:
        values = [c["case_number"], c["pattern_title"], c["true_class"],
                  f'[{c["sample_id"]}](<{c["image_path"]}>)',
                  *(f'{c["correct_seeds"][m]}/3' for m in MODELS)]
        candidate_lines.append("| " + " | ".join(map(str, values)) + " |")
    candidate_lines += ["", "Visual inspection of these six candidates is documented in case_details.json and ERROR_ANALYSIS.md in the example gallery folder.", ""]
    (output / "CANDIDATE_CASES.md").write_text("\n".join(candidate_lines), encoding="utf-8")

    save_json(output / "verification.json", {
        "status": "passed", "prediction_source_hashes_checked": len(scores["inputs"]),
        "sample_ids_match_test_manifest": True, "candidate_images_checked": len(cases),
        "visual_review_performed": False,
    })
    save_json(ROOT / "logs/M5_ERROR_ANALYSIS_STATUS.json", {
        "status": "in_progress", "error_dir": str(output.relative_to(ROOT)),
        "source_prediction_run": str(RUN.relative_to(ROOT)),
        "step1_counts_complete": True, "step2_candidates_prepared": True,
        "candidate_count": 6, "step2_visual_review_complete": False,
        "gallery_complete": False, "final_written_analysis_complete": False,
        "next_step": "visually review the six candidate images",
        "updated_at": datetime.now().astimezone().isoformat(),
    })
    status_path = ROOT / "logs/M5_METRICS_STATUS.json"
    status = json.loads(status_path.read_text())
    status.update({"error_statistics_completed": True,
                   "error_analysis_dir": str(output.relative_to(ROOT)),
                   "error_case_analysis_performed": False,
                   "next_step": "visually review the six candidate images",
                   "updated_at": datetime.now().astimezone().isoformat()})
    save_json(status_path, status)
    print(json.dumps({"error_dir": str(output),
                      "seed_counts": [{"seed": x["seed"], **x["pattern_counts"],
                                       "errors": x["model_errors"]} for x in per_seed],
                      "candidates": [{k: c[k] for k in ("sample_id", "pattern_title", "correct_seeds")} for c in cases]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
