import argparse
import hashlib
import json
from pathlib import Path

import xlsxwriter


MODELS = ("A", "F", "B")
GENERATORS = ("glide", "Midjourney", "wukong")
SEEDS = (42, 123, 2026)
METRICS = ("auroc", "accuracy", "precision", "recall")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, output_dir):
    source = Path(source).resolve()
    source_hash = digest(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["status"] != "complete" or len(data["per_generator"]) != 27:
        raise ValueError("Expected the complete 27-row score table")
    rows = {(x["model"], x["seed"], x["generator"]): (i + 2, x)
            for i, x in enumerate(data["per_generator"])}
    expected_keys = {(m, s, g) for m in MODELS for s in SEEDS for g in GENERATORS}
    if set(rows) != expected_keys:
        raise ValueError("Unexpected model, seed, or generator rows")

    seed_rows = {(x["model"], x["seed"]): x for x in data["per_seed_stability"]}
    stability = [seed_rows[model, seed] for model in MODELS for seed in SEEDS]
    model_summary = {x["model"]: x for x in data["model_summary"]}
    stability_keys = ("glide_auroc", "Midjourney_auroc", "wukong_auroc", "macro_auroc",
                      "cross_generator_sd", "worst_generator_auroc", "auroc_range")
    model_keys = ("glide_auroc_mean", "Midjourney_auroc_mean", "wukong_auroc_mean", "macro_auroc_mean",
                  "macro_auroc_seed_sd", "cross_generator_sd_mean", "worst_generator_auroc_mean",
                  "worst_generator_auroc_seed_sd")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "M5_results_comparison.xlsx"
    report_path = output_dir / "workbook_validation.json"
    if target.exists() or report_path.exists():
        raise FileExistsError("Use a new output folder to preserve existing results")

    with xlsxwriter.Workbook(target) as book:
        book.set_properties({"title": "M5 Results", "author": "", "company": ""})
        book.set_calc_mode("auto")
        summary = book.add_worksheet("Model comparison")
        detail = book.add_worksheet("Run metrics")
        base = {"font_name": "Arial", "font_size": 11, "font_color": "#202B38", "valign": "vcenter"}

        def fmt(**extra):
            return book.add_format(dict(base, **extra))

        normal = fmt()
        number = fmt(num_format="0.0000", align="right")
        percentage = fmt(num_format="0.00%", align="right")
        count = fmt(num_format="#,##0", align="right")
        seed_fmt = fmt(num_format="0", align="center")
        note = fmt(font_color="#526676")
        header = fmt(bg_color="#284B63", font_color="#FFFFFF", bold=True,
                     align="center", text_wrap=True, left=1, right=1, left_color="#FFFFFF", right_color="#FFFFFF")
        for sheet in (summary, detail):
            sheet.hide_gridlines(2)
            sheet.set_default_row(18.75)
        summary.set_tab_color("#284B63")
        summary.set_column_pixels("A:A", 78, normal)
        summary.set_column_pixels("B:D", 126, normal)
        summary.set_column_pixels("E:I", 124, normal)
        summary.set_column_pixels("J:J", 116, normal)
        detail.set_column_pixels("A:A", 70, normal)
        detail.set_column_pixels("B:B", 74, seed_fmt)
        detail.set_column_pixels("C:C", 124, normal)
        detail.set_column_pixels("D:I", 104, count)
        detail.set_column_pixels("J:M", 102, normal)
        detail.freeze_panes(1, 3)

        def heading(sheet, row, labels):
            sheet.set_row(row - 1, 48)
            sheet.write_row(row - 1, 0, labels, header)

        summary.write("A2", "Model test results", fmt(font_size=16, bold=True, bottom=1, bottom_color="#93A7B5"))
        for col in range(1, 10):
            summary.write_blank(1, col, None, fmt(bottom=1, bottom_color="#93A7B5"))
        summary.write("A3", "Results averaged across three seeds.", fmt(italic=True, font_color="#526676"))
        heading(summary, 5, ["Model", "glide\nAUROC", "Midjourney\nAUROC", "wukong\nAUROC", "Mean\nAUROC", "Seed SD\n(mean AUROC)", "Cross-generator\nSD", "Worst-generator\nAUROC", "Seed SD\n(worst AUROC)"])
        for index, model in enumerate(MODELS):
            row, first = 6 + index, 34 + 3 * index
            last = first + 2
            summary.write(f"A{row}", model)
            formulas = [f"=AVERAGE('Run metrics'!{c}{first}:{c}{last})" for c in "CDEF"]
            formulas += [f"=STDEV('Run metrics'!F{first}:F{last})", f"=AVERAGE('Run metrics'!G{first}:G{last})",
                         f"=AVERAGE('Run metrics'!H{first}:H{last})", f"=STDEV('Run metrics'!H{first}:H{last})"]
            for col, (formula, key) in enumerate(zip(formulas, model_keys), 1):
                summary.write_formula(row - 1, col, formula, number, model_summary[model][key])
        for row, text in enumerate((
            "Mean AUROC: average across generators within each seed, then across seeds. Worst AUROC: minimum per seed, then average.",
            "Cross-generator SD: population SD across generators (ddof=0), averaged across seeds. Lower is more consistent.",
            "Seed SD: sample SD across three seeds (ddof=1)."), 10):
            summary.write(f"A{row}", text, note)
        summary.write("A15", "Metrics by generator", fmt(font_size=14, bold=True))
        heading(summary, 17, ["Model", "Generator", "AUROC\nmean", "AUROC\nseed SD", "Accuracy\nmean", "Accuracy\nseed SD", "Precision\nmean", "Precision\nseed SD", "Recall\nmean", "Recall\nseed SD"])
        for index, group in enumerate(data["generator_summary"]):
            row = 18 + index
            summary.write_row(row - 1, 0, [group["model"], group["generator"]])
            refs = [rows[group["model"], seed, group["generator"]][0] for seed in SEEDS]
            formulas = []
            for col in "JKLM":
                args = ",".join(f"'Run metrics'!{col}{r}" for r in refs)
                formulas += [f'=IF(COUNT({args})=3,{fn}({args}),"n.a.")' for fn in ("AVERAGE", "STDEV")]
            values = [group[f"{metric}_{suffix}"] for metric in METRICS for suffix in ("mean", "seed_sd")]
            for col, (formula, value) in enumerate(zip(formulas, values), 2):
                value = value if value is not None else "n.a."
                summary.write_formula(row - 1, col, formula, number if col < 4 else percentage, value)

        heading(detail, 1, ["Model", "Seed", "Generator", "Real\nimages", "AI\nimages", "TN", "FP", "FN", "TP", "AUROC", "Accuracy", "Precision", "Recall"])
        for index, x in enumerate(data["per_generator"]):
            row = 2 + index
            detail.write_row(row - 1, 0, [x[k] for k in ("model", "seed", "generator", "n_real", "n_fake", "tn", "fp", "fn", "tp")])
            detail.write_number(row - 1, 9, x["auroc"], number)
            formulas = [f"=(F{row}+I{row})/SUM(F{row}:I{row})",
                        f'=IF(SUM(G{row},I{row})=0,"n.a.",I{row}/SUM(G{row},I{row}))', f"=I{row}/SUM(H{row}:I{row})"]
            for col, (formula, metric) in enumerate(zip(formulas, METRICS[1:]), 10):
                value = x[metric] if x[metric] is not None else "n.a."
                detail.write_formula(row - 1, col, formula, percentage, value)
        detail.write("A31", "Cross-generator statistics by seed", fmt(font_size=14, bold=True))
        heading(detail, 33, ["Model", "Seed", "glide\nAUROC", "Midjourney\nAUROC", "wukong\nAUROC", "Mean\nAUROC", "Cross-generator\nSD", "Worst-generator\nAUROC", "AUROC\nrange"])
        for index, x in enumerate(stability):
            row = 34 + index
            detail.write_row(row - 1, 0, [x["model"], x["seed"]])
            formulas = [f"=J{rows[x['model'], x['seed'], g][0]}" for g in GENERATORS]
            formulas += [f"=AVERAGE(C{row}:E{row})", f"=STDEVP(C{row}:E{row})", f"=MIN(C{row}:E{row})", f"=MAX(C{row}:E{row})-MIN(C{row}:E{row})"]
            for col, (formula, key) in enumerate(zip(formulas, stability_keys), 2):
                detail.write_formula(row - 1, col, formula, number, x[key])
        detail.write("A45", "Test settings", fmt(font_size=14, bold=True))
        notes = [
            "Per generator: 2,500 AI + the same 1,166 real images = 3,666. AI=1; real=0; threshold=0.5.",
            "AUROC uses continuous probabilities. Other metrics use the classification threshold.",
            "Undefined precision is shown as n.a.",
            "Methods and sources: SCORING.md."]
        for row, text in enumerate(notes, 46):
            detail.write(f"A{row}", text, note)
    report = {"status": "complete",
              "source_scores_sha256": source_hash, "workbook_sha256": digest(target), "workbook": target.name,
              "cached_values_source": source.name, "native_excel_engine_checked": False,
              "calculation_note": "Cached results from scores.json; Excel recalculates on open."}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export M5 scores to Excel")
    parser.add_argument("scores", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.scores, args.output_dir), ensure_ascii=False, indent=2))
