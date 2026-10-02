import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from category_guidance import (DSMGeometryInput, ConfidenceFeatureFusion,
                               CategoryGuidedPAN, calibrate_categories,
                               category_confidence, normalize_dsm)
from segmentation_loss import segmentation_loss
import UNetFormer_MMSAM as model_module
from MedSAM.models.sam.modeling.image_encoder import ImageEncoderViT


torch.set_num_threads(2)


def small_sam(args, checkpoint):
    """Real SAM/LoRA encoder at reduced width/depth; no pretrained file needed."""
    sam = nn.Module()
    sam.image_encoder = ImageEncoderViT(
        args, img_size=64, patch_size=16, embed_dim=32, depth=1,
        num_heads=4, out_chans=256, window_size=4,
    )
    return sam


def make_model(**options):
    window_size = options.pop('window_size', 2)
    with patch.object(model_module.cfg, "parse_args",
                      return_value=SimpleNamespace(mod="sam_lora", mid_dim=4)), \
            patch.dict(model_module.sam_model_registry, {"vit_l": small_sam}):
        return model_module.UNetFormer(window_size=window_size, **options)


class GeometryTests(unittest.TestCase):
    def test_flat_and_invalid_tiles_are_finite(self):
        for tile in [np.ones((5, 7)), np.full((5, 7), np.nan)]:
            normalized = normalize_dsm(tile)
            np.testing.assert_array_equal(normalized, np.zeros_like(tile))
            result = DSMGeometryInput()(torch.from_numpy(normalized)[None])
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(result[:, 1:].abs().max().item(), 0)
        mixed = normalize_dsm(np.array([[np.nan, 10], [np.inf, 20]]))
        np.testing.assert_array_equal(mixed, [[0, 0], [0, 1]])

    def test_sobel_direction_and_flip(self):
        height = torch.linspace(0, 1, 9)[None, None, :].expand(1, 7, 9)
        geometry = DSMGeometryInput()
        output = geometry(height)
        torch.testing.assert_close(output[:, :1], height[:, None])
        self.assertTrue((output[:, 1, :, 1:-1] > 0).all())
        torch.testing.assert_close(output[:, 2], torch.zeros_like(height), atol=1e-7, rtol=0)
        flipped = geometry(height.flip(-1))
        torch.testing.assert_close(flipped[:, 1], -output[:, 1].flip(-1), atol=1e-7, rtol=0)

    def test_disabled_geometry_matches_repeated_dsm(self):
        dsm = torch.rand(2, 1, 5, 7)
        torch.testing.assert_close(DSMGeometryInput(False)(dsm), dsm.repeat(1, 3, 1, 1))
        with self.assertRaises(ValueError):
            DSMGeometryInput()(dsm.repeat(1, 3, 1, 1))


class GuidanceTests(unittest.TestCase):
    def test_confidence_limits(self):
        uniform = torch.full((1, 6, 3, 5), 1 / 6)
        self.assertEqual(category_confidence(uniform).abs().max().item(), 0)
        certain = torch.zeros_like(uniform)
        certain[:, 2] = 1
        torch.testing.assert_close(category_confidence(certain), torch.ones(1, 1, 3, 5))

    def test_calibration_fallback_and_reliable_modality(self):
        logits = torch.zeros(1, 6, 3, 5, requires_grad=True)
        categories, share = calibrate_categories(logits, logits)
        torch.testing.assert_close(share, torch.full_like(share, 0.5))
        self.assertFalse(categories.requires_grad)
        reliable = torch.zeros_like(logits)
        reliable[:, 2] = 12
        categories, share = calibrate_categories(reliable, logits)
        self.assertTrue((share[:, 2] > 0.99).all())
        self.assertTrue((categories[:, 2] > 0.99).all())
        torch.testing.assert_close(categories.sum(1), torch.ones(1, 3, 5))
        inverse, inverse_share = calibrate_categories(logits, reliable)
        torch.testing.assert_close(categories, inverse)
        torch.testing.assert_close(share, 1 - inverse_share)

    def test_fusion_preserves_baseline_when_uncertain(self):
        fusion = ConfidenceFeatureFusion(8, 6)
        rgb, dsm = torch.randn(2, 8, 5, 7), torch.randn(2, 8, 5, 7)
        share = torch.full((2, 6, 5, 7), 0.5)
        torch.testing.assert_close(fusion(rgb, dsm, share), rgb + dsm)
        selected = fusion(torch.ones_like(rgb), torch.zeros_like(dsm), torch.ones_like(share))
        self.assertTrue((selected > 1).all())
        self.assertTrue((selected < 2).all())

    def test_pan_preserves_probability_and_accepts_uneven_scales(self):
        pan = CategoryGuidedPAN(8, 6)
        maps = [torch.randn(2, 6, h, w).softmax(1)
                for h, w in [(17, 19), (9, 10), (5, 5), (3, 3)]]
        guidance = pan(maps)
        for source, result in zip(maps, guidance):
            self.assertEqual(source.shape, result.shape)
            torch.testing.assert_close(result.sum(1), torch.ones_like(result[:, 0]))
        feature = torch.randn(2, 8, 17, 19)
        uniform = torch.full((2, 6, 17, 19), 1 / 6)
        torch.testing.assert_close(pan.gate(feature, uniform, 0), feature)
        pan.gate(feature, guidance[0], 0).square().mean().backward()
        self.assertGreater(pan.mix_logits.grad.abs().sum().item(), 0)


class LossTests(unittest.TestCase):
    def test_auxiliary_losses_train_each_prediction(self):
        logits = torch.randn(2, 6, 16, 16, requires_grad=True)
        rgb = [torch.randn(2, 6, size, size, requires_grad=True) for size in (4, 2)]
        dsm = [torch.randn(2, 6, 4, 4, requires_grad=True)]
        pan = [torch.randn(2, 6, 4, 4, requires_grad=True)]
        target = torch.randint(0, 6, (2, 16, 16))
        target[:, 0] = 6
        output = logits, dict(rgb=rgb, dsm=dsm, pan=pan)
        loss, terms = segmentation_loss(output, target)
        loss.backward()
        for prediction in [logits] + rgb + dsm + pan:
            self.assertGreater(prediction.grad.abs().sum().item(), 0)
        expected = terms['main'] + 0.4 * (terms['rgb'] + terms['dsm']) + 0.2 * terms['pan']
        torch.testing.assert_close(loss, expected)

    def test_all_ignored_and_disabled_auxiliary(self):
        logits = torch.randn(2, 6, 8, 8, requires_grad=True)
        auxiliary = torch.randn_like(logits, requires_grad=True)
        loss, _ = segmentation_loss((logits, auxiliary), torch.full((2, 8, 8), 6))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        target = torch.randint(0, 6, (2, 8, 8))
        baseline, _ = segmentation_loss((logits, auxiliary), target, category_weight=0,
                                       pan_weight=0, decoder_weight=0)
        torch.testing.assert_close(baseline, torch.nn.functional.cross_entropy(logits, target))


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_training_gradients_and_eval_checkpoint_round_trip(self):
        model = make_model()
        rgb, dsm = torch.rand(2, 3, 64, 64), torch.rand(2, 64, 64)
        output = model(rgb, dsm)
        self.assertEqual(output[0].shape, (2, 6, 64, 64))
        self.assertEqual(len(output[1]['rgb']), 4)
        loss, _ = segmentation_loss(output, torch.randint(0, 6, (2, 64, 64)))
        loss.backward()
        for heads in (model.rgb_category_heads, model.dsm_category_heads, model.pan_category_heads):
            for head in heads:
                gradient = head.head[-1].weight.grad
                self.assertIsNotNone(gradient)
                self.assertGreater(gradient.abs().sum().item(), 0)
        for fusion in (model.fusion1, model.fusion2, model.fusion3):
            self.assertGreater(fusion.confidence_fusion.channel_categories.grad.abs().sum().item(), 0)
        self.assertGreater(model.category_pan.mix_logits.grad.abs().sum().item(), 0)
        for name, param in model.image_encoder.named_parameters():
            if 'lora_' not in name:
                self.assertIsNone(param.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.image_encoder.named_parameters() if 'lora_' in n))
        torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-3).step()
        model.eval()
        with torch.no_grad():
            expected = model(rgb, dsm)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = make_model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(rgb, dsm), expected)

    def test_ablation_combinations(self):
        for supervision, fusion, pan, geometry in [
                (False, False, False, False), (False, False, False, True),
                (True, False, False, False), (True, True, False, False),
                (True, False, True, False), (True, True, True, False)]:
            with self.subTest(options=(supervision, fusion, pan, geometry)):
                model = make_model(use_category_supervision=supervision,
                                   use_confidence_fusion=fusion, use_category_pan=pan,
                                   use_dsm_geometry=geometry).eval()
                with torch.no_grad():
                    result = model(torch.rand(1, 3, 64, 64), torch.ones(1, 64, 64))
                self.assertEqual(result.shape, (1, 6, 64, 64))
                self.assertTrue(torch.isfinite(result).all())
                if not supervision:
                    self.assertFalse(any('category' in k for k in model.state_dict()))
        with self.assertRaises(ValueError):
            make_model(use_category_supervision=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_default_crop_and_window_on_cuda(self):
        model = make_model(window_size=8).cuda().train()
        rgb = torch.rand(1, 3, 256, 256, device='cuda')
        dsm = torch.rand(1, 256, 256, device='cuda')
        output = model(rgb, dsm)
        self.assertEqual(output[0].shape, (1, 6, 256, 256))
        loss, _ = segmentation_loss(output, torch.randint(0, 6, (1, 256, 256), device='cuda'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.rgb_category_heads[0].head[-1].weight.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_mixed_precision_training(self):
        model = make_model().cuda().train()
        rgb, dsm = torch.rand(2, 3, 64, 64, device='cuda'), torch.rand(2, 64, 64, device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(rgb, dsm)
            loss, _ = segmentation_loss(output, torch.randint(0, 6, (2, 64, 64), device='cuda'))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())


if __name__ == '__main__':
    unittest.main()
