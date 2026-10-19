import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import evaluate


ARCHITECTURE = "m3_frozen_frequency_concat_v1"
evaluate.ARCHITECTURES["B2"] = ("baseline_b_2", ARCHITECTURE, 768)
original_load_model = evaluate.load_model
original_validate_inputs = evaluate.validate_inputs


def validate_inputs(config, models, seeds):
    rows, selected, metadata = original_validate_inputs(config, models, seeds)
    if metadata["test_csv_sha256"] != config["expected_test_csv_sha256"]:
        raise ValueError("The original M5 test manifest has changed")
    return rows, selected, metadata


def load_model(item, config):
    if item["model"] != "B2":
        return original_load_model(item, config)

    code_dir = evaluate.project_path(config["code_dir"])
    for name, expected in config["b2_source_sha256"].items():
        if evaluate.sha256(code_dir / name) != expected:
            raise ValueError(f"B2 source checksum mismatch: {name}")
    path = evaluate.project_path(item["checkpoint"])
    if evaluate.sha256(path) != item["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")

    import torch
    from torchvision.models import resnet18
    from models.baseline_b_2 import BaselineB2

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    identity = (checkpoint.get("format_version"), checkpoint.get("model_name"),
                checkpoint.get("architecture"))
    if identity != (2, "baseline_b_2", ARCHITECTURE):
        raise ValueError("Expected a Baseline B2 checkpoint")
    if checkpoint["config"]["seed"] != item["seed"]:
        raise ValueError("Checkpoint seed mismatch")
    if checkpoint["config"]["learning_rate"] != config["training_learning_rate"]:
        raise ValueError("Checkpoint learning rate mismatch")
    if checkpoint["selection_metric"] != "validation_auroc" or checkpoint["epoch"] != checkpoint["best_epoch"]:
        raise ValueError("Expected the best validation-AUROC checkpoint")

    def checkpoint_resnet18(*args, **kwargs):
        kwargs["weights"] = None
        return resnet18(*args, **kwargs)

    with ExitStack() as stack:
        for module in ("models.spatial_encoder", "models.frequency_encoder_frozen"):
            stack.enter_context(patch(f"{module}.resnet18", checkpoint_resnet18))
        model = BaselineB2()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    metadata = {
        **item, "model_name": "baseline_b_2", "architecture": ARCHITECTURE,
        "best_epoch": checkpoint["best_epoch"],
        "validation_auroc_at_selection": checkpoint["selected_checkpoint_validation_AUROC"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "strict_load_passed": True, "network_downloads": 0,
        "b2_entrypoint_sha256": evaluate.sha256(__file__),
        "frequency_projection": "randomly initialized, then frozen during training",
    }
    return model, metadata


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(x == "--config" or x.startswith("--config=") for x in arguments):
        arguments += ["--config", str(Path(__file__).with_name("evaluation_b2_config.json"))]
    (evaluate.PROJECT_ROOT / "logs" / "tmp").mkdir(parents=True, exist_ok=True)
    evaluate.load_model = load_model
    evaluate.validate_inputs = validate_inputs
    return evaluate.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
