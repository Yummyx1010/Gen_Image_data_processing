import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/m5_test_20261008_124810_081d31'
SCORES = ROOT / 'results/m5_scores_20261008_184008_9498f1'
R_SCRIPT = Path(__file__).with_suffix('.R')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-output-dir', type=Path)
    args = parser.parse_args()
    scores = json.loads((SCORES / 'scores.json').read_text())
    report = json.loads((RUN / 'run_report.json').read_text())
    assert scores['status'] == report['status'] == 'complete'
    assert report['samples_per_model'] == 8666 and len(report['models']) == 9
    sources = {}
    for item in scores['inputs']:
        path = ROOT / item['path']
        assert sha256(path) == item['sha256'], f'Original input changed: {path}'
        sources[path] = item['sha256']
    for name in ['scores.json', 'per_generator_metrics.csv', 'per_seed_stability.csv']:
        path = SCORES / name
        sources[path] = sha256(path)

    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    if args.resume_output_dir:
        output = args.resume_output_dir.resolve()
        assert output.parent == (ROOT / 'results').resolve()
        assert output.name.startswith('m5_figures_original_')
        assert output.is_dir() and not (output / 'figure_manifest.json').exists()
    else:
        output = ROOT / 'results' / f'm5_figures_original_{stamp}_{uuid.uuid4().hex[:6]}'
        output.mkdir()
    result = subprocess.run(['/usr/local/bin/Rscript', '--vanilla', str(R_SCRIPT),
                             str(RUN), str(SCORES), str(output)],
                            cwd=output, text=True, capture_output=True)
    (output / 'plot_console.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Plot creation failed: {result.stdout}{result.stderr}')
    image_names = ['01_AUROC_by_generator', '02_mean_and_consistency', '03_ROC_by_generator']
    outputs = [{'file': f'{name}.png', 'sha256': sha256(output / f'{name}.png')}
               for name in image_names]
    roc_checks = read_csv(output / 'roc_checks.csv')
    max_error = max(abs(float(r['calculated_auroc']) - float(r['recorded_auroc'])) for r in roc_checks)
    manifest = {
        'status': 'awaiting_visual_review', 'created_at': datetime.now().astimezone().isoformat(),
        'scope': 'original_full_test_all_nine_checkpoints',
        'models': ['A', 'F', 'B'], 'seeds': [42, 123, 2026], 'images_per_model': 8666,
        'generators': ['glide', 'Midjourney', 'wukong'],
        'fake_per_generator': 2500, 'shared_real_images': 1166,
        'aggregate_definition': 'Compute generator metrics per seed, then aggregate across seeds.',
        'cross_generator_sd_ddof': 0, 'seed_errorbar_sd_ddof': 1,
        'roc_display': 'Thin lines: individual seeds. Thick lines: pointwise mean on FPR grid. Legend AUC: mean of exact per-seed AUROC.',
        'roc_curve_checks': 27, 'max_roc_auroc_error': max_error,
        'source_files_recorded': len(sources),
        'inputs': [{'path': str(p.relative_to(ROOT)), 'sha256': digest} for p, digest in sources.items()],
        'builders': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in [Path(__file__), R_SCRIPT]],
        'outputs': outputs,
    }
    (output / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    note = '''# Original full-test figures

The figures use the original A, F and B checkpoints with seeds 42, 123 and 2026: nine checkpoints in total. Every checkpoint was evaluated on all 8,666 original test images.

1. **01_AUROC_by_generator.png** compares AUROC on the three unseen generators. Small points show individual seeds; diamonds and error bars show the seed mean and sample SD.
2. **02_mean_and_consistency.png** shows mean AUROC (higher is better) and cross-generator SD (lower is more consistent). Both statistics are calculated within each seed, then summarised across seeds. Error bars on the SD panel show variation between training seeds.
3. **03_ROC_by_generator.png** shows ROC curves for each unseen generator. False positive rate is the proportion of real images classified as AI; true positive rate is the proportion of AI images correctly detected. Thin lines show individual seeds and thick lines show interpolated mean curves. Legend AUROCs are means of exact per-seed AUROCs, not approximate areas under the thick curves.

Each generator evaluation uses 2,500 AI images and the same 1,166 real images. Seed error bars are not confidence intervals; no significance test was performed.
'''
    (output / 'FIGURES.md').write_text(note, encoding='utf-8')
    progress = {'status': 'awaiting_visual_review', 'figure_dir': str(output.relative_to(ROOT)),
                'source_prediction_run': str(RUN.relative_to(ROOT)),
                'figures_generated': 3, 'error_case_analysis_performed': False,
                'updated_at': datetime.now().astimezone().isoformat()}
    (ROOT / 'logs/M5_PLOTS_STATUS.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2))
    print(json.dumps({'figure_dir': str(output), 'plots': image_names, 'roc_checks': 27,
                      'max_roc_auroc_error': max_error},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
