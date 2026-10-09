import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/m5_test_20261008_124810_081d31"
SCORES = ROOT / "results/m5_scores_20261008_184008_9498f1"
SEEDS = (42, 123, 2026)
MODELS = ("A", "F", "B")
REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
OBSERVATIONS = {
    "glide/GLIDE_1000_200_01_109_glide_00035.jpg": ("Coral-like textured close-up",
        "Coral-like structures appear against a blue-green background, with dense curved textures in orange-yellow areas."),
    "Midjourney/112_midjourney_23.jpg": ("Seashell on a sandy beach",
        "A close-up of a seashell on sand. The foreground is relatively sharp, while the sea and sky are blurred."),
    "wukong/10_wukong_image72.jpg": ("Two birds on a branch",
        "Two brightly coloured birds perch on a branch against a mostly blurred green background."),
    "Nature/ILSVRC2012_val_00004645.JPEG": ("A bed with white curtains",
        "White curtains surround a wooden bed. The area behind the bed is brightly lit and the foreground is darker."),
    "glide/GLIDE_1000_200_00_013_glide_00106.jpg": ("A bird among bare branches",
        "A small grey-white bird perches among thin branches. The background has bright areas and the image is generally blurred."),
    "Midjourney/0_midjourney_161.jpg": ("Close-up of a yellow fish",
        "A close-up of a yellow fish head with a clearly visible eye and mouth against a blurred dark-green background."),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def describe_results(auroc_a, auroc_b, sd_a, sd_b):
    improved = []
    if auroc_b > auroc_a:
        improved.append("mean AUROC")
    if sd_b < sd_a:
        improved.append("cross-generator consistency")
    outcome = "improved " + " and ".join(improved) if improved else "did not improve mean AUROC or cross-generator consistency"
    return f"The frequency branch {outcome} under the evaluated settings."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-dir", type=Path)
    args = parser.parse_args()
    step1 = args.counts_dir
    if step1 is None:
        status = json.loads((ROOT / "logs/M5_METRICS_STATUS.json").read_text())
        step1 = ROOT / status["error_analysis_dir"]
    candidates_path = step1 / "candidate_cases.json"
    counts_path = step1 / "error_counts.json"
    candidates = json.loads(candidates_path.read_text())
    counts = json.loads(counts_path.read_text())
    cases = candidates["cases"]
    assert len(cases) == 6
    protected = {ROOT / item["path"]: item["sha256"] for item in counts["inputs"]}
    for path in (candidates_path, counts_path, SCORES / "model_summary.csv"):
        protected[path] = digest(path)
    for path, sha in protected.items():
        assert digest(path) == sha, f"Input changed: {path}"

    image_sizes = {}
    for case in cases:
        image_path = Path(case["image_path"])
        with Image.open(image_path) as im:
            image_sizes[case["case_number"]] = list(im.size)

    output = ROOT / "results" / f"m5_error_examples_original_{uuid.uuid4().hex[:8]}"
    output.mkdir()
    now = datetime.now().astimezone().isoformat()
    reviewed_cases = []
    for case in cases:
        item = dict(case)
        title, observation = OBSERVATIONS[case["sample_id"]]
        outcome = "; ".join(f"{model}: {case['correct_seeds'][model]}/3 correct" for model in MODELS) + "."
        item.update({"visually_reviewed": True, "visual_review_method": "visual inspection of the original local image",
                     "original_pixel_size": image_sizes[case["case_number"]],
                     "display_description_en": title, "visual_observation_en": observation,
                     "prediction_observation_en": outcome})
        reviewed_cases.append(item)
    save_json(output / "case_details.json", {
        "status": "six_examples_visually_reviewed",
        "scope": "original_full_test_all_nine_checkpoints",
        "created_at": now, "threshold": 0.5, "seeds": SEEDS,
        "selection_rule": candidates["selection_rule"],
        "purpose": candidates["purpose"], "cases": reviewed_cases,
    })

    width, height = 2400, 1880
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    fonts = {name: ImageFont.truetype(str(path), size) for name, path, size in (
        ("title", BOLD, 56), ("subtitle", REGULAR, 31), ("column", BOLD, 37),
        ("panel", BOLD, 32), ("caption", REGULAR, 29),
        ("count", BOLD, 39), ("legend", REGULAR, 29), ("foot", REGULAR, 29))}
    text_boxes = []

    def text(x, y, value, font, fill="#18222E", center=False):
        f = fonts[font]
        bbox = draw.textbbox((0, 0), value, font=f)
        line_width = bbox[2] - bbox[0]
        if center:
            x -= line_width / 2
        actual = draw.textbbox((x, y), value, font=f)
        assert 0 <= actual[0] < actual[2] <= width and 0 <= actual[1] < actual[3] <= height, (value, actual)
        draw.text((x, y), value, font=f, fill=fill)
        text_boxes.append({"text": value, "bbox": list(actual)})

    text(60, 30, "Error examples across three training seeds", "title")
    text(60, 110, "Original A / F / B checkpoints | seeds 42, 123, 2026 | AI probability threshold = 0.5", "subtitle", "#4B5969")
    left, gap, col_width = 60, 30, 740
    headers = ("A correct / B wrong", "A wrong / B correct", "A and B wrong")
    for col, label in enumerate(headers):
        x = left + col * (col_width + gap)
        draw.rounded_rectangle((x, 178, x + col_width, 252), radius=12, fill="#EAF0F6")
        text(x + col_width / 2, 191, label, "column", center=True)
    by_number = {c["case_number"]: c for c in reviewed_cases}
    displayed = []
    for col, numbers in enumerate(((1, 2), (3, 4), (5, 6))):
        for row, number in enumerate(numbers):
            case = by_number[number]
            x = left + col * (col_width + gap)
            y = 278 + row * 758
            draw.rounded_rectangle((x, y, x + col_width, y + 734), radius=14,
                                   outline="#CDD6E0", width=2, fill="white")
            true_class = "Real" if case["true_label"] == 0 else "AI"
            text(x + 24, y + 16, f"Case {number} | {case['generator']} | True: {true_class}", "panel")
            text(x + 24, y + 61, case["display_description_en"], "caption", "#4B5969")
            box = (x + 24, y + 110, x + col_width - 24, y + 590)
            draw.rectangle(box, fill="#F3F5F7")
            with Image.open(case["image_path"]) as source:
                scaled = ImageOps.contain(source.convert("RGB"), (int(box[2] - box[0]), int(box[3] - box[1])), Image.Resampling.LANCZOS)
                px = int((box[0] + box[2] - scaled.width) / 2)
                py = int((box[1] + box[3] - scaled.height) / 2)
                canvas.paste(scaled, (px, py))
            text(x + 24, y + 608, "Correct classifications across 3 seeds:", "legend", "#4B5969")
            for index, model in enumerate(MODELS):
                n = case["correct_seeds"][model]
                color = "#176B58" if n == 3 else ("#97481C" if n == 0 else "#4B5969")
                text(x + 24 + index * 230, y + 653, f"{model}: {n}/3", "count", color)
            displayed.append({"case_number": number, "image_box": box,
                              "displayed_image_size": list(scaled.size), "correct_seeds": case["correct_seeds"]})

    text(60, 1822, "A: spatial only     F: frequency only     B: spatial + frequency", "foot", "#4B5969")
    image_path = output / "04_error_examples.png"
    canvas.save(image_path, dpi=(300, 300))

    summaries = {r["model"]: r for r in read_csv(SCORES / "model_summary.csv")}
    auroc_a = float(summaries["A"]["macro_auroc_mean"])
    auroc_b = float(summaries["B"]["macro_auroc_mean"])
    sd_a = float(summaries["A"]["cross_generator_sd_mean"])
    sd_b = float(summaries["B"]["cross_generator_sd_mean"])
    comparison = describe_results(auroc_a, auroc_b, sd_a, sd_b)
    lines = ["# Error analysis", "",
             "A is spatial-only, F is frequency-only and B combines spatial and frequency features. Results use seeds 42, 123 and 2026 on 8,666 test images.", "",
             "## Case selection", "",
             "The six images were selected by source coverage and filename order, with two images per A/B error pattern shared by all three seeds.", "",
             "P(AI) >= 0.5 is classified as AI. The gallery reports correct classifications out of three seeds.", "",
             "## Visual observations", "",
             "| Case | Source / true class | Visible content | A correct | F correct | B correct |",
             "|---|---|---|---:|---:|---:|"]
    for case in reviewed_cases:
        values = [case["case_number"], f"{case['generator']} / {case['true_class']}",
                  case["visual_observation_en"], *(f"{case['correct_seeds'][m]}/3" for m in MODELS)]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "## Findings", "",
              "In cases 1 and 2, A detects the AI images while B classifies them as real in all three seeds. In cases 3 and 4, B correctly classifies an AI bird image and a real bedroom image that A misclassifies. Both A and B classify the AI images in cases 5 and 6 as real.", "",
              "In case 5, F is correct in two seeds while B is wrong in all three. F and B are trained separately; B uses frequency features rather than F's final classification.", "",
              "## Full-test results", "",
              f"Mean AUROC is {auroc_a:.4f} for A and {auroc_b:.4f} for B; cross-generator SD is {sd_a:.4f} for A and {sd_b:.4f} for B. {comparison}", "",
              "Mean AUROC and cross-generator SD are calculated within each seed and then averaged across seeds.", "",
              "## Supporting data", "",
              "Sample IDs and all 54 per-seed probabilities and predictions are in CASE_PREDICTIONS.md and case_details.json.", ""]
    (output / "ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Per-seed predictions for the six cases", "",
             "Probabilities are P(AI), with a fixed threshold of 0.5. Ground truth comes from the original test manifest. Probabilities and model predictions are reported separately.", ""]
    for case in reviewed_cases:
        lines += [f"## Case {case['case_number']}: {case['sample_id']}", "",
                  f"Original image: [view image](<{case['image_path']}>). True class: {case['true_class']}.", "",
                  "| Seed | A: P(AI) / prediction | F: P(AI) / prediction | B: P(AI) / prediction |",
                  "|---|---|---|---|"]
        for seed in SEEDS:
            values = [str(seed)]
            for model in MODELS:
                p = case["predictions"][model][str(seed)]
                label = "AI" if p["pred_label"] else "Real"
                verdict = "correct" if p["correct"] else "wrong"
                values.append(f"{p['probability_ai']:.4f} / {label} ({verdict})")
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    (output / "CASE_PREDICTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(output / "figure_manifest.json", {
        "status": "awaiting_figure_visual_review", "created_at": now,
        "source_prediction_run": str(RUN.relative_to(ROOT)),
        "source_candidates": str(candidates_path.relative_to(ROOT)),
        "models": MODELS, "seeds": SEEDS, "full_test_images": 8666,
        "example_count": 6, "individual_image_visual_review_complete": True,
        "display": {"width": width, "height": height, "crop": False,
                    "enhancement": False, "aspect_ratio_preserved": True,
                    "case_order_by_column": [[1, 2], [3, 4], [5, 6]], "panels": displayed},
        "text_boxes": text_boxes,
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha} for path, sha in protected.items()],
        "builder": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))},
        "outputs": [{"file": p.name, "sha256": digest(p)} for p in sorted(output.iterdir()) if p.is_file() and not p.name.startswith("._")],
        "verification": {"case_predictions_source": str(candidates_path.relative_to(ROOT)),
                         "source_hashes_checked": len(protected), "all_text_inside_canvas": True,
                         "all_six_original_images_readable": True},
    })
    print(json.dumps({"output_dir": str(output), "figure": str(image_path),
                      "case_predictions": len(cases) * len(MODELS) * len(SEEDS),
                      "figure_visual_review_pending": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
