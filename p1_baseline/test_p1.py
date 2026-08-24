from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from p1_baseline.config import Config, load_config, scientific_config_hash
from p1_baseline.data import (
    EpochPermutationSampler,
    EpochWeightedSampler,
    build_train_sampler,
    manifest_hash,
)
from p1_baseline.data import BoneAgeDataset, load_manifest
from p1_baseline.metrics import compute_metrics
from p1_baseline.model import build_model
from p1_baseline.preflight import sampling_report
from p1_baseline.trainer import CHECKPOINT_KEYS, Trainer, atomic_torch_save
from p1_baseline.trainer import gaussian_label_distribution, restore_rng_state, rng_state


class P1UnitTests(unittest.TestCase):
    def test_operational_change_does_not_change_scientific_hash(self):
        cfg = Config()
        changed = replace(cfg, num_workers=7, log_every_steps=1)
        self.assertEqual(scientific_config_hash(cfg), scientific_config_hash(changed))

    def test_scientific_change_changes_hash(self):
        cfg = Config()
        self.assertNotEqual(scientific_config_hash(cfg), scientific_config_hash(replace(cfg, learning_rate=1e-3)))

    def test_sampler_resume_suffix(self):
        full = list(EpochPermutationSampler(20, 42, 3, 0))
        resumed = list(EpochPermutationSampler(20, 42, 3, 7))
        self.assertEqual(full[7:], resumed)

    def test_weighted_sampler_is_deterministic_and_resume_is_suffix(self):
        weights = [0.75, 1.0, 1.5, 0.9]
        full = list(EpochWeightedSampler(weights, 20, 42, 3, 0))
        repeated = list(EpochWeightedSampler(weights, 20, 42, 3, 0))
        resumed = list(EpochWeightedSampler(weights, 20, 42, 3, 7))
        self.assertEqual(full, repeated)
        self.assertEqual(full[7:], resumed)
        self.assertEqual(len(full), 20)

    def test_weighted_sampler_favors_larger_weights(self):
        sampled = list(EpochWeightedSampler([0.01, 1.0], 1000, 42, 0, 0))
        self.assertGreater(sampled.count(1), 950)

    def test_sampler_factory_selects_locked_strategy_and_resume_length(self):
        rows = [
            {"sample_weight": "0.75000000"},
            {"sample_weight": "1.50000000"},
            {"sample_weight": "1.00000000"},
        ]
        permutation = build_train_sampler(rows, "permutation", 42, 2, 1)
        weighted = build_train_sampler(rows, "manifest_weighted", 42, 2, 1)
        self.assertIsInstance(permutation, EpochPermutationSampler)
        self.assertIsInstance(weighted, EpochWeightedSampler)
        self.assertEqual(len(permutation), 2)
        self.assertEqual(len(weighted), 2)

    def test_sampler_factory_rejects_missing_or_invalid_manifest_weights(self):
        with self.assertRaisesRegex(ValueError, "sample_weight"):
            build_train_sampler([{}], "manifest_weighted", 42, 0, 0)
        with self.assertRaisesRegex(ValueError, "sample_weight"):
            build_train_sampler(
                [{"sample_weight": "nan"}], "manifest_weighted", 42, 0, 0
            )

    def test_trainer_loader_uses_manifest_weighted_sampler(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = replace(
            Config(), sampling_strategy="manifest_weighted", num_workers=0, batch_size=2
        )
        trainer.train_rows = [
            {
                "split": "train",
                "image_id": "1",
                "bone_age_months": "30.0",
                "sex": "F",
                "image_path": "unused.png",
                "sample_weight": "0.75000000",
            },
            {
                "split": "train",
                "image_id": "2",
                "bone_age_months": "150.0",
                "sex": "M",
                "image_path": "unused.png",
                "sample_weight": "1.50000000",
            },
        ]
        trainer.val_rows = []
        trainer.device = torch.device("cpu")
        trainer.epoch = 0
        loader = trainer._loader(train=True, start_index=0)
        self.assertIsInstance(loader.sampler, EpochWeightedSampler)

    def test_preflight_sampling_report_exposes_weighted_sequence_health(self):
        rows = [
            {"sample_weight": "0.75000000"},
            {"sample_weight": "1.50000000"},
            {"sample_weight": "1.00000000"},
        ]
        report = sampling_report(rows, "manifest_weighted", seed=42)
        self.assertEqual(report["strategy"], "manifest_weighted")
        self.assertEqual(report["num_samples"], 3)
        self.assertEqual(report["weight_min"], 0.75)
        self.assertEqual(report["weight_max"], 1.5)
        self.assertGreaterEqual(report["unique_indices"], 1)

    def test_baseline_manifest_hash_remains_locked(self):
        rows = load_manifest("p0_audit/outputs/train_manifest.csv", "train")
        self.assertEqual(
            manifest_hash(rows),
            "7328667e6822ab074d442155e33eba89606861bb13bbf804be8a1138c78f5285",
        )

    def test_curated_manifest_hash_changes_when_weight_changes(self):
        base = {
            "split": "train",
            "image_id": "1",
            "bone_age_months": "30.0",
            "sex": "F",
            "sha256": "a" * 64,
            "sample_weight": "1.00000000",
            "age_bin": "000-059",
            "sex_age_stratum": "F_000-059",
            "audit_status": "KEEP",
            "audit_reason": "",
        }
        changed = dict(base, sample_weight="1.10000000")
        self.assertNotEqual(manifest_hash([base]), manifest_hash([changed]))

    def test_config_rejects_unknown_sampling_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text('[training]\nsampling_strategy = "mystery"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sampling_strategy"):
                load_config(path)

    def test_checkpoint_contract_includes_partial_epoch_loss(self):
        self.assertIn("epoch_loss_sum", CHECKPOINT_KEYS)
        self.assertIn("epoch_loss_count", CHECKPOINT_KEYS)

    def test_augmentation_is_deterministic_for_epoch_and_id(self):
        rows = [dict(load_manifest("p0_audit/outputs/train_manifest.csv", "train")[0])]
        rows[0]["image_path"] = str(
            Path("data/goc/boneage-training-dataset/boneage-training-dataset")
            / f"{rows[0]['image_id']}.png"
        )
        kwargs = dict(
            rows=rows, image_size=64, target_mean=127.32, target_std=41.18,
            train=True, epoch=2, seed=42, augmentation="light",
            horizontal_flip_probability=0.5, rotation_degrees=7.0,
            translation_fraction=0.03, scale_min=0.95, scale_max=1.05,
            brightness_delta=0.1, contrast_delta=0.1, gamma_min=0.9, gamma_max=1.1,
        )
        first = BoneAgeDataset(**kwargs)[0]["image"]
        second = BoneAgeDataset(**kwargs)[0]["image"]
        self.assertTrue(torch.equal(first, second))

    def test_prediction_warning_thresholds_are_nested(self):
        cfg = Config()
        self.assertLess(cfg.warning_prediction_hard_min, cfg.warning_prediction_soft_min)
        self.assertLess(cfg.warning_prediction_soft_max, cfg.warning_prediction_hard_max)

    def test_preprocessing_changes_scientific_hash(self):
        cfg = Config()
        self.assertNotEqual(
            scientific_config_hash(cfg),
            scientific_config_hash(replace(cfg, preprocessing="official_mask_v1")),
        )

    def test_p9_recipe_options_change_scientific_hash(self):
        cfg = Config()
        candidate = replace(
            cfg, preprocessing="deeplasia_mask_histogram_v1",
            image_normalization="per_image_zscore", augmentation="deeplasia_fancy",
            regression_loss="mae", shear_degrees=10.0, clahe_probability=0.67,
        )
        self.assertNotEqual(scientific_config_hash(cfg), scientific_config_hash(candidate))

    def test_sex_mode_changes_scientific_hash(self):
        cfg = Config()
        self.assertNotEqual(
            scientific_config_hash(cfg),
            scientific_config_hash(replace(cfg, sex_mode="none")),
        )
        self.assertNotEqual(
            scientific_config_hash(cfg),
            scientific_config_hash(replace(cfg, sex_mode="dual_output")),
        )

    def test_e1_embedding_default_is_backward_compatible(self):
        torch.manual_seed(123)
        default_model = build_model("smoke_cnn", False, 4, 8, 0.0).eval()
        torch.manual_seed(123)
        explicit_model = build_model(
            "smoke_cnn", False, 4, 8, 0.0, sex_mode="embedding"
        ).eval()
        self.assertEqual(
            list(default_model.state_dict()), list(explicit_model.state_dict())
        )
        for key, value in default_model.state_dict().items():
            self.assertTrue(torch.equal(value, explicit_model.state_dict()[key]))
        image = torch.randn(2, 3, 32, 32)
        sex = torch.tensor([[0.0], [1.0]])
        with torch.inference_mode():
            self.assertTrue(torch.equal(
                default_model(image, sex), explicit_model(image, sex)
            ))

    def test_e0_image_only_ignores_sex(self):
        model = build_model(
            "smoke_cnn", False, 4, 8, 0.0, sex_mode="none"
        ).eval()
        image = torch.randn(1, 3, 32, 32).repeat(2, 1, 1, 1)
        sex = torch.tensor([[0.0], [1.0]])
        with torch.inference_mode():
            output = model(image, sex)
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_e2_dual_output_routes_by_sex(self):
        model = build_model(
            "smoke_cnn", False, 4, 8, 0.0, sex_mode="dual_output"
        ).eval()
        with torch.no_grad():
            model.female_head.weight.zero_()
            model.female_head.bias.fill_(-2.0)
            model.male_head.weight.zero_()
            model.male_head.bias.fill_(3.0)
        image = torch.randn(4, 3, 32, 32)
        sex = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        with torch.inference_mode():
            output = model(image, sex)
        self.assertTrue(torch.equal(output, torch.tensor([-2.0, 3.0, -2.0, 3.0])))

    def test_e2_only_selected_head_and_backbone_receive_gradients(self):
        model = build_model(
            "smoke_cnn", False, 4, 8, 0.0, sex_mode="dual_output"
        )
        image = torch.randn(3, 3, 32, 32)

        model(image, torch.zeros(3, 1)).sum().backward()
        self.assertGreater(model.female_head.bias.grad.abs().sum().item(), 0.0)
        self.assertEqual(model.male_head.bias.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.features[0].weight.grad.abs().sum().item(), 0.0)

        model.zero_grad(set_to_none=True)
        model(image, torch.ones(3, 1)).sum().backward()
        self.assertEqual(model.female_head.bias.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.male_head.bias.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.features[0].weight.grad.abs().sum().item(), 0.0)

    def test_e2_state_dict_round_trip(self):
        source = build_model(
            "smoke_cnn", False, 4, 8, 0.0, sex_mode="dual_output"
        ).eval()
        restored = build_model(
            "smoke_cnn", False, 4, 8, 0.0, sex_mode="dual_output"
        ).eval()
        restored.load_state_dict(source.state_dict(), strict=True)
        image = torch.randn(2, 3, 32, 32)
        sex = torch.tensor([[0.0], [1.0]])
        with torch.inference_mode():
            self.assertTrue(torch.equal(source(image, sex), restored(image, sex)))

    def test_e2_rejects_invalid_sex_shape(self):
        model = build_model(
            "smoke_cnn", False, 4, 8, 0.0, sex_mode="dual_output"
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, 3, 32, 32), torch.tensor([0.0, 1.0]))

    def test_new_sex_modes_reject_unsupported_backbones(self):
        with self.assertRaisesRegex(ValueError, "chỉ hỗ trợ"):
            build_model(
                "efficientnet_b0", False, 4, 8, 0.0, sex_mode="dual_output"
            )

    def test_convnextv2_tiny_forward_contract(self):
        model = build_model("convnextv2_tiny", False, 16, 32, 0.0).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 64, 64), torch.zeros(1, 1))
        self.assertEqual(tuple(output.shape), (1,))
        self.assertEqual(model.backbone.num_features, 768)

    def test_efficientnet_b0_forward_contract(self):
        model = build_model("efficientnet_b0", False, 32, 256, 0.2).eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 64, 64), torch.zeros(2, 1))
        self.assertEqual(tuple(output.shape), (2,))
        self.assertEqual(model.features[0][0].in_channels, 1)
        self.assertEqual(model.features[-1][0].out_channels, 1280)

    def test_multiscale_forward_contract_and_identity_initialization(self):
        model = build_model("convnext_tiny_multiscale", False, 16, 32, 0.0).eval()
        image = torch.randn(2, 3, 64, 64)
        sex = torch.tensor([[0.0], [1.0]])
        with torch.inference_mode():
            stages, fused = model.forward_features(image)
            output = model(image, sex)
        self.assertEqual([tuple(stage.shape) for stage in stages], [(2, 96), (2, 192), (2, 384), (2, 768)])
        self.assertEqual(tuple(fused.shape), (2, 768))
        self.assertTrue(torch.allclose(fused, stages[-1], atol=1e-6, rtol=1e-6))
        self.assertEqual(tuple(output.shape), (2,))

    def test_label_distribution_forward_and_gradients(self):
        model = build_model("convnext_tiny_ldl", False, 16, 32, 0.0, 229)
        output = model(torch.randn(2, 3, 64, 64), torch.tensor([[0.0], [1.0]]))
        self.assertEqual(tuple(output["regression"].shape), (2,))
        self.assertEqual(tuple(output["distribution_logits"].shape), (2, 229))
        target = gaussian_label_distribution(torch.tensor([10.0, 200.0]), 229, 2.0)
        loss = output["regression"].square().mean() - 0.2 * (
            target * torch.log_softmax(output["distribution_logits"], dim=1)
        ).sum(dim=1).mean()
        loss.backward()
        self.assertIsNotNone(model.regressor[-1].weight.grad)
        self.assertIsNotNone(model.distribution_head[-1].weight.grad)

    def test_gaussian_label_distribution_is_normalized_and_centered(self):
        target = gaussian_label_distribution(torch.tensor([0.0, 57.0, 228.0]), 229, 2.0)
        self.assertTrue(torch.allclose(target.sum(dim=1), torch.ones(3), atol=1e-6))
        self.assertEqual(target.argmax(dim=1).tolist(), [0, 57, 228])

    def test_rng_restore_accepts_cpu_state(self):
        state = rng_state()
        restore_rng_state(state)
        self.assertEqual(state["torch_cpu"].device.type, "cpu")

    def test_metrics(self):
        records = [
            {"target_months": 10.0, "prediction_months": 12.0, "sex": "M"},
            {"target_months": 20.0, "prediction_months": 16.0, "sex": "F"},
        ]
        result = compute_metrics(records, [0, 60, 120, 180, 229])
        self.assertAlmostEqual(result["mae"], 3.0)
        self.assertAlmostEqual(result["rmse"], (10.0) ** 0.5)
        self.assertEqual(result["prediction_std_by_sex"], {"F": 0.0, "M": 0.0})

    def test_checkpoint_key_validation(self):
        state = {key: None for key in CHECKPOINT_KEYS}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.ckpt"
            atomic_torch_save(path, state)
            loaded = torch.load(path, weights_only=False)
            self.assertEqual(set(loaded), CHECKPOINT_KEYS)


if __name__ == "__main__":
    unittest.main()
