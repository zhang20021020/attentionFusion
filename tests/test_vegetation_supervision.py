import io
import math
import unittest

import torch

from checkpoint_utils import checkpoint_state_dict
from segmentation_loss import segmentation_loss
from vegetation_supervision import VegetationAuxHead, vegetation_cross_entropy
from test_category_guidance import make_model
from test_dsm_geometry import NEW_OPTIONS


OPTIONS = dict(NEW_OPTIONS, dsm_geometry_windows=(17, 45), use_vegetation_aux=True)


class VegetationLossTests(unittest.TestCase):
    def test_label_mapping_mask_and_mean_over_only_vegetation(self):
        target = torch.tensor([[[0, 2, 3, 4], [5, 6, 1, 2]]])
        logits = torch.tensor([[[[99., math.log(3), 0., 99.],
                                 [99., 99., 99., 0.]],
                                [[-99., 0., math.log(3), -99.],
                                 [-99., -99., -99., math.log(3)]]]], requires_grad=True)
        # Two correct predictions with p=3/4, one incorrect prediction with p=1/4.
        loss = vegetation_cross_entropy(logits, target)
        self.assertAlmostEqual(loss.item(), (2 * math.log(4 / 3) + math.log(4)) / 3, places=6)
        loss.backward()
        valid = ((target == 2) | (target == 3))[:, None].expand_as(logits)
        self.assertEqual(logits.grad[~valid].abs().sum().item(), 0)
        self.assertTrue((logits.grad[valid].abs() > 0).all())

    def test_empty_vegetation_and_all_ignored_batches_have_zero_finite_gradients(self):
        for label in (0, 5, 6):
            with self.subTest(label=label):
                logits = torch.randn(2, 2, 4, 5, requires_grad=True)
                loss = vegetation_cross_entropy(logits, torch.full((2, 16, 20), label))
                self.assertEqual(loss.item(), 0)
                loss.backward()
                torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_upsampling_single_class_and_custom_ignore(self):
        logits = torch.tensor([[[[math.log(3)]], [[0.]]]], requires_grad=True)
        target = torch.full((1, 7, 9), 2)
        self.assertAlmostEqual(vegetation_cross_entropy(logits, target).item(),
                               math.log(4 / 3), places=6)
        self.assertEqual(vegetation_cross_entropy(logits, target, ignore_index=2).item(), 0)
        target.fill_(3)
        self.assertAlmostEqual(vegetation_cross_entropy(logits, target).item(),
                               math.log(4), places=6)

    def test_total_loss_weight_and_disabled_backward_compatibility(self):
        logits = torch.randn(2, 6, 16, 16, requires_grad=True)
        vegetation = torch.randn(2, 2, 2, 2, requires_grad=True)
        target = torch.randint(0, 7, (2, 16, 16))
        output = logits, {'vegetation': vegetation}
        loss, terms = segmentation_loss(output, target, category_weight=0, pan_weight=0,
                                         vegetation_weight=.02)
        torch.testing.assert_close(loss, terms['main'] + .02 * terms['vegetation'])
        loss.backward()
        self.assertGreater(vegetation.grad.abs().sum().item(), 0)
        disabled, terms = segmentation_loss(output, target, category_weight=0, pan_weight=0)
        torch.testing.assert_close(disabled, terms['main'])
        self.assertEqual(terms['vegetation'].item(), 0)
        with self.assertRaisesRegex(ValueError, 'requires training output'):
            segmentation_loss(logits, target, vegetation_weight=.02)
        with self.assertRaises(ValueError):
            segmentation_loss(output, target, vegetation_weight=-.02)


class VegetationIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_head_uses_corrected_res2y_and_its_loss_trains_geometry_after_zero_init(self):
        model = make_model(**OPTIONS).train()
        seen = {}
        handles = [
            model.dsm_geometry_branch.register_forward_hook(
                lambda module, inputs, output: seen.update(corrected=output)),
            model.vegetation_aux_head.register_forward_pre_hook(
                lambda module, inputs: seen.update(head_input=inputs[0])),
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        rgb, dsm = torch.rand(2, 3, 64, 64), torch.rand(2, 64, 64)
        target = torch.full((2, 64, 64), 2)
        target[:, :, 32:] = 3
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(rgb, dsm)
            self.assertIs(seen['head_input'], seen['corrected'][1])
            self.assertEqual(output[1]['vegetation'].shape, (2, 2, 8, 8))
            total, terms = segmentation_loss(output, target, category_weight=.1,
                                              pan_weight=.05, vegetation_weight=.02)
            self.assertTrue(torch.isfinite(total))
            # Isolate the new loss: its gradients must reach the geometry branch.
            terms['vegetation'].backward()
            self.assertGreater(model.vegetation_aux_head.head[-1].weight.grad.abs().sum().item(), 0)
            injection = model.dsm_geometry_branch.injections[0]
            self.assertGreater(injection.projection.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(model.dsm_geometry_branch.encoder.stem[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(injection.gate[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(injection.strength.grad.abs().item(), 0)
            # The new loss directly supervises 1/8 only, not the 1/16 injection.
            self.assertIsNone(model.dsm_geometry_branch.injections[1].projection.weight.grad)
            self.assertTrue(all(p.grad is None for n, p in model.image_encoder.named_parameters()
                                if 'lora_' not in n))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.grad is not None))
            optimizer.step()
        for handle in handles:
            handle.remove()

    def test_eval_skips_head_and_common_initial_weights_and_predictions_are_unchanged(self):
        old_options = dict(OPTIONS, use_vegetation_aux=False)
        baseline = make_model(**old_options).eval()
        torch.manual_seed(42)
        enhanced = make_model(**OPTIONS).eval()
        enhanced_state = enhanced.state_dict()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, enhanced_state[key], atol=0, rtol=0)
        self.assertFalse(any(k.startswith('vegetation_aux_head.') for k in baseline.state_dict()))
        calls = []
        handle = enhanced.vegetation_aux_head.register_forward_hook(
            lambda module, inputs, output: calls.append(output))
        rgb, dsm = torch.rand(1, 3, 64, 64), torch.rand(1, 64, 64)
        with torch.no_grad():
            expected = baseline(rgb, dsm, mode='Test')
            actual = enhanced(rgb, dsm, mode='Test')
        self.assertEqual(actual.shape, (1, 6, 64, 64))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(calls, [])
        handle.remove()

    def test_vegetation_supervision_without_category_guidance_still_returns_auxiliary(self):
        options = dict(OPTIONS, use_category_supervision=False,
                       use_confidence_fusion=False, use_category_pan=False)
        model = make_model(**options).train()
        output = model(torch.rand(2, 3, 64, 64), torch.rand(2, 64, 64))
        self.assertIn('vegetation', output[1])
        loss, _ = segmentation_loss(output, torch.full((2, 64, 64), 2),
                                     category_weight=0, pan_weight=0, vegetation_weight=.02)
        loss.backward()
        self.assertGreater(model.vegetation_aux_head.head[-1].weight.grad.abs().sum().item(), 0)

    def test_requires_geometry_and_isprs_label_space(self):
        with self.assertRaisesRegex(ValueError, 'geometry_branch'):
            make_model(**dict(OPTIONS, use_dsm_geometry_branch=False))
        with self.assertRaisesRegex(ValueError, 'class IDs'):
            make_model(**dict(OPTIONS, num_classes=3))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_head_and_full_size_targets(self):
        head = VegetationAuxHead().cuda().train()
        features = torch.randn(2, 256, 32, 32, device='cuda', requires_grad=True)
        target = torch.randint(0, 7, (2, 256, 256), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = vegetation_cross_entropy(head(features), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(features.grad).all())


class VegetationCheckpointTests(unittest.TestCase):
    def test_old_metadata_and_state_load_strictly_with_head_disabled(self):
        old_options = dict(NEW_OPTIONS, dsm_geometry_windows=(17, 45))
        model = make_model(**old_options)
        checkpoint = dict(model_state_dict=model.state_dict(), model_options=old_options, num_classes=6)
        current_options = dict(old_options, use_vegetation_aux=False)
        restored = make_model(**current_options)
        restored.load_state_dict(checkpoint_state_dict(checkpoint, current_options, 6), strict=True)
        with self.assertRaises(ValueError):
            checkpoint_state_dict(checkpoint, OPTIONS, 6)

    def test_new_checkpoint_roundtrip_and_wrong_flag_rejected(self):
        model = make_model(**OPTIONS).eval()
        checkpoint = dict(model_state_dict=model.state_dict(), model_options=OPTIONS,
                          loss_options={'vegetation_weight': .02}, num_classes=6)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        # This buffer was created above. PyTorch 1.13's restricted loader cannot
        # decode float metadata (vegetation_weight); ordinary loading can.
        loaded = torch.load(buffer, weights_only=False)
        restored = make_model(**OPTIONS).eval()
        restored.load_state_dict(checkpoint_state_dict(loaded, OPTIONS, 6), strict=True)
        rgb, dsm = torch.rand(1, 3, 64, 64), torch.rand(1, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(restored(rgb, dsm), model(rgb, dsm), atol=0, rtol=0)
        self.assertEqual(loaded['loss_options']['vegetation_weight'], .02)
        with self.assertRaises(ValueError):
            checkpoint_state_dict(loaded, dict(OPTIONS, use_vegetation_aux=False), 6)
        with self.assertRaises(ValueError):
            checkpoint_state_dict({}, {'use_vegetation_aux': True}, 6)


if __name__ == '__main__':
    unittest.main()
