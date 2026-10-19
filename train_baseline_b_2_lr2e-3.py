"""Run Baseline B_2 with seed supplied by CLI and learning rate fixed at 0.002.

This launcher preserves the byte identity of the original B_2 training script.
It validates that the new protocol differs only in learning rate and phase.
"""

import argparse
import json

from train_baseline_b_2 import DEFAULT_PROTOCOL, load_protocol, make_config, run_training
from training.config import resolve_path
from training.data import inspect_splits


PROTOCOL_PATH = "configs/baseline_b_2_lr2e-3.json"


def load_lr2e3_protocol(path=PROTOCOL_PATH):
    reference = load_protocol(DEFAULT_PROTOCOL)
    expected = {
        **reference,
        "phase": "baseline_b_2_lr2e-3",
        "learning_rate_candidates": [0.002],
        "learning_rate": 0.002,
    }
    with resolve_path(path).open(encoding="utf-8-sig") as stream:
        protocol = json.load(stream)
    if protocol != expected:
        differences = sorted(key for key in protocol.keys() | expected.keys()
                             if protocol.get(key) != expected.get(key))
        raise ValueError(f"Expected only the B_2 learning-rate change: {differences}")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--inspect-data", action="store_true")
    args = parser.parse_args()
    protocol = load_lr2e3_protocol()
    if args.inspect_data:
        config = make_config(args.seed, args.data_root, protocol)
        print(json.dumps(inspect_splits(config), indent=2))
        return
    run_training(args.seed, args.data_root, protocol)


if __name__ == "__main__":
    main()
