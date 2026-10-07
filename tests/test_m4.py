"""Offline integration tests. Random test weights never constitute model results."""

import copy
import csv
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from models.baseline_b import BaselineB
from models.baseline_a_adapter import FrozenBaselineA
from models.frequency_encoder import FrequencyEncoder
from training.config import ROOT, TrainConfig
from training.data import inspect_splits, make_loader
from training.smoke import FrequencyStub, offline_spatial_initialization
from training.train import (PREDICTION_FIELDS, build_model, export_predictions,
                                load_checkpoint, run_epoch, seed_everything, train)


class SmallSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 512))

    def forward(self, images):
        return self.layers(images)


class M4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        seed_everything(42)
        self.config = TrainConfig(baseline="B", smoke=True, device="cpu", epochs=2, batch_size=4,
                                  frequency_factory="training.smoke:FrequencyStub", frequency_input="rgb01")

    def test_b_architecture_and_single_sample(self):
        model = BaselineB(FrequencyStub(), "rgb01", SmallSpatial())
        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.in_features, 768)
        self.assertEqual(model.classifier.out_features, 1)
        self.assertEqual(model(torch.rand(1, 3, 32, 32)).shape, (1, 1))

    def test_real_frequency_encoder_integration_and_gradients(self):
        model = BaselineB(FrequencyEncoder(feature_dim=256), "normalized", SmallSpatial())
        rgb = torch.rand(2, 3, 32, 32)
        normalized = (rgb - model.mean) / model.std
        logits = model(normalized)
        self.assertEqual(logits.shape, (2, 1))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.spatial_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.frequency_encoder.parameters()))
        self.assertTrue(model.classifier.weight.grad is not None)

    @unittest.skipUnless(os.environ.get("M4_DATA_ROOT"), "set M4_DATA_ROOT for the real-data integration check")
    def test_real_data_full_baseline_b_integration(self):
        config = replace(TrainConfig.load("configs/m4_baseline_b.json"),
                         data_root=os.environ["M4_DATA_ROOT"], batch_size=4, device="cpu")
        batch = next(iter(make_loader(config, "train")))
        images = batch["image"]
        labels = batch["label"].to(dtype=torch.float32).reshape(-1, 1)
        model = build_model(config)
        model.train()
        logits = model(images)
        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(logits.shape, (4, 1))
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.spatial_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.frequency_encoder.parameters()))
        self.assertTrue(model.classifier.weight.grad is not None)

    def test_frozen_spatial_weights_and_batchnorm_frequency_and_head_update(self):
        model = BaselineB(FrequencyStub(), "rgb01", SmallSpatial())
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        run_epoch(model, make_loader(self.config, "train"), torch.device("cpu"), optimizer)
        self.assertFalse(model.spatial_encoder.training)
        self.assertTrue(all(p.grad is None for p in model.spatial_encoder.parameters()))
        for name, value in model.state_dict().items():
            if name.startswith("spatial_encoder."):
                self.assertTrue(torch.equal(before[name], value), name)
        for prefix in ("frequency_encoder.", "classifier."):
            self.assertTrue(any(not torch.equal(before[name], value)
                                for name, value in model.named_parameters() if name.startswith(prefix)))

    def test_a_reuses_m2_state_keys_and_linear_head(self):
        from models.baseline_a import BaselineA
        with offline_spatial_initialization():
            original, adapted = BaselineA(), FrozenBaselineA()
        adapted.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(list(original.state_dict()), list(adapted.state_dict()))
        adapted.train()
        self.assertFalse(adapted.spatial_encoder.training)
        self.assertEqual(sum(p.numel() for p in adapted.parameters() if p.requires_grad), 513)
        self.assertEqual(adapted(torch.rand(1, 3, 64, 64)).shape, (1, 1))
        original.eval()
        image = torch.rand(2, 3, 64, 64)
        self.assertTrue(torch.equal(original(image), adapted(image)))

    def test_frequency_dimensions_are_enforced(self):
        wrong = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 128))
        model = BaselineB(wrong, "normalized", SmallSpatial())
        with self.assertRaisesRegex(ValueError, "256"):
            model(torch.rand(2, 3, 32, 32))

    def test_detached_frequency_is_rejected(self):
        class Detached(FrequencyStub):
            def forward(self, images):
                return super().forward(images).detach()
        model = BaselineB(Detached(), "normalized", SmallSpatial())
        with self.assertRaisesRegex(RuntimeError, "Detached"):
            model(torch.rand(2, 3, 32, 32))

    def test_rgb01_is_derived_from_the_same_normalized_image(self):
        class Recorder(FrequencyStub):
            def forward(self, images):
                self.received = images.detach().clone()
                return super().forward(images)
        frequency = Recorder()
        model = BaselineB(frequency, "rgb01", SmallSpatial())
        rgb = torch.rand(2, 3, 32, 32)
        model((rgb - model.mean) / model.std)
        self.assertTrue(torch.allclose(rgb, frequency.received, atol=1e-6))

    def test_formal_configs_use_real_frequency_encoder_and_no_silent_fallback(self):
        from train_final import load_protocol
        for path in (ROOT / "configs").glob("result_compare*_b.json"):
            with self.subTest(protocol=path.name):
                protocol_b = load_protocol(path)
                paired_a = path.with_name(path.name.replace("_b.json", "_a.json"))
                paired_f = path.with_name(path.name.replace("_b.json", "_frequency.json"))
                for paired in (paired_a, paired_f):
                    self.assertTrue(paired.exists(), f"Missing paired protocol: {paired}")
                    other = load_protocol(paired)
                    for name in ("learning_rate", "max_epochs", "batch_size", "weight_decay",
                                 "grad_clip", "sampler", "loss", "optimizer", "selection_metric"):
                        self.assertEqual(other[name], protocol_b[name])
                self.assertEqual(protocol_b["architecture"], "m3_frequency_concat_v1")
                self.assertEqual(protocol_b["model_source"], [
                    "models/spatial_encoder.py", "models/frequency_encoder_pretrained.py", "models/baseline_b.py"
                ])
                self.assertIs(protocol_b["freeze_frequency_backbone"], False)
                self.assertNotIn("frequency_checkpoint", protocol_b)
                self.assertIsNone(protocol_b["hidden_dim"])
                self.assertIsNone(protocol_b["dropout"])
        config_b = replace(self.config, smoke=False, frequency_factory="missing_m3:Encoder")
        with self.assertRaisesRegex(RuntimeError, "Frequency encoder is unavailable"):
            build_model(config_b)
        with self.assertRaisesRegex(ValueError, "stub"):
            replace(self.config, smoke=False).validate(require_data=False)

    def test_supplied_splits(self):
        report = inspect_splits(TrainConfig())
        self.assertEqual([report[s]["samples"] for s in ("train", "validation", "test")],
                         [10996, 3666, 8666])

    def test_overlapping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for split in ("train", "validation", "test"):
                (Path(directory) / f"{split}.csv").write_text(
                    "image_path,label,generator,split\n"
                    f"Nature/shared.jpg,0,Nature,{split}\n{split}/fake.jpg,1,{split},{split}\n",
                    encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                inspect_splits(TrainConfig(split_dir=directory))

    def test_m1_dataset_and_preprocessing_reused_without_rewriting_csv(self):
        from dataset import GenImageDataset, transform
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Nature").mkdir()
            Image.new("RGB", (300, 280), (50, 100, 160)).save(root / "Nature/sample.jpg")
            path = root / "train.csv"
            original = "image_path,label,generator,split\nNature\\sample.jpg,0,Nature,train\n"
            path.write_text(original, encoding="utf-8")
            loader = make_loader(TrainConfig(data_root=directory, split_dir=directory), "train")
            self.assertIsInstance(loader.dataset, GenImageDataset)
            self.assertIs(loader.dataset.transform, transform)
            self.assertEqual(next(iter(loader))["image"].shape, (1, 3, 224, 224))
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_checkpoint_resume_and_seven_column_prediction_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.config, output_dir=directory)
            with patch("training.train.make_loader", wraps=make_loader) as loader_spy:
                continuous = train(config)
                self.assertEqual([call.args[1] for call in loader_spy.call_args_list], ["train", "validation"])
            interrupted = train(replace(config, epochs=1))
            resumed = train(config, resume=interrupted / "last.pt")
            full, resumed_state = load_checkpoint(continuous / "last.pt"), load_checkpoint(resumed / "last.pt")
            self.assertEqual(full["history"], resumed_state["history"])
            for name, tensor in full["model_state_dict"].items():
                self.assertTrue(torch.equal(tensor, resumed_state["model_state_dict"][name]), name)
            predictions = export_predictions(resumed / "best.pt")
            with predictions.open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, PREDICTION_FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({row["sample_id"] for row in rows}), 4)
            self.assertTrue(all(row["model_name"] == "BaselineB" and row["seed"] == "42" for row in rows))
            self.assertTrue(all(0 <= float(row["pred_probability"]) <= 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
