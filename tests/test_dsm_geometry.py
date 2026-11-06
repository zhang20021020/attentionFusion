import io
import unittest

import torch
from torch import nn

from checkpoint_utils import checkpoint_state_dict
from dsm_geometry import DSMGeometryDescriptors, DSMGeometryBranch
from segmentation_loss import segmentation_loss
from test_category_guidance import make_model


R2_OPTIONS = dict(use_category_supervision=True, use_confidence_fusion=True,
                  use_category_pan=True, use_dsm_geometry=False,
                  use_n2_category_gate=False)
NEW_OPTIONS = dict(R2_OPTIONS, use_dsm_geometry_branch=True, dsm_geometry_windows=(9, 25))


class DescriptorTests(unittest.TestCase):
    def test_five_channels_match_a_known_local_neighborhood(self):
        dsm = torch.zeros(1, 11, 11)
        dsm[0, 5, 5] = 1
        features = DSMGeometryDescriptors((3, 5))(dsm)
        self.assertEqual(features.shape, (1, 5, 11, 11))
        torch.testing.assert_close(features[:, 0], dsm)
        for i, k in enumerate((3, 5)):
            mean = 1 / (k * k)
            self.assertAlmostEqual(features[0, i + 1, 5, 5].item(), 1 - mean, places=6)
            self.assertAlmostEqual(features[0, i + 3, 5, 5].item(),
                                   (mean - mean * mean + 1e-6) ** .5, places=6)
        self.assertLess(features[0, 1, 5, 4].item(), 0)

    def test_flat_borders_and_rotation_do_not_create_geometry_edges(self):
        descriptors = DSMGeometryDescriptors()
        flat = descriptors(torch.ones(1, 1, 17, 21))
        torch.testing.assert_close(flat[:, 1:3], torch.zeros_like(flat[:, 1:3]))
        torch.testing.assert_close(flat[:, 3:], torch.full_like(flat[:, 3:], .001))
        torch.manual_seed(7)
        dsm = torch.rand(1, 31, 37)
        rotated = descriptors(dsm.rot90(1, (-2, -1)))
        torch.testing.assert_close(rotated, descriptors(dsm).rot90(1, (-2, -1)),
                                   atol=3e-6, rtol=1e-4)

    def test_statistics_stay_float32_in_autocast_and_backward_is_finite(self):
        dsm = torch.ones(1, 17, 21, requires_grad=True)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            features = DSMGeometryDescriptors()(dsm)
        self.assertEqual(features.dtype, torch.float32)
        features.mean().backward()
        self.assertTrue(torch.isfinite(dsm.grad).all())

    def test_configuration_validation(self):
        for windows in [(9,), (9, 24), (25, 9), (9, 9), (1, 9), (9., 25)]:
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                DSMGeometryDescriptors(windows)
        with self.assertRaises(ValueError):
            DSMGeometryDescriptors()(torch.rand(1, 3, 8, 8))


class GeometryBranchTests(unittest.TestCase):
    def test_default_crop_two_scale_injection_and_learning_after_zero_init(self):
        torch.manual_seed(42)
        branch = DSMGeometryBranch()
        dsm = torch.rand(1, 256, 256)
        optical = [torch.randn(1, 256, s, s) for s in (64, 32, 16, 8)]
        original = [torch.randn_like(x) for x in optical]
        optimizer = torch.optim.SGD(branch.parameters(), lr=.1)
        for step in range(2):
            optimizer.zero_grad()
            output = branch(dsm, optical, original)
            self.assertIs(output[0], original[0])
            self.assertIs(output[3], original[3])
            for i in (1, 2):
                self.assertEqual(output[i].shape, original[i].shape)
                if step == 0:
                    torch.testing.assert_close(output[i], original[i], atol=0, rtol=0)
                else:
                    self.assertGreater((output[i] - original[i]).abs().max().item(), 0)
            loss = sum((output[i] - torch.ones_like(output[i])).square().mean() for i in (1, 2))
            loss.backward()
            for injection in branch.injections:
                self.assertGreater(injection.projection.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(branch.encoder.stem[0].weight.grad.abs().sum().item(), 0)
                for injection in branch.injections:
                    self.assertGreater(injection.gate[0].weight.grad.abs().sum().item(), 0)
                    self.assertGreater(injection.strength.grad.abs().item(), 0)
            for p in branch.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()

    def test_optical_guidance_changes_nonzero_residual(self):
        branch = DSMGeometryBranch().eval()
        for injection in branch.injections:
            nn.init.normal_(injection.projection.weight, std=.02)
        dsm = torch.rand(1, 64, 64)
        optical = [torch.randn(1, 256, s, s) for s in (16, 8, 4, 2)]
        original = [torch.randn_like(x) for x in optical]
        with torch.no_grad():
            first = branch(dsm, optical, original)
            second = branch(dsm, [-x for x in optical], original)
        for i in (1, 2):
            self.assertGreater((first[i] - second[i]).abs().max().item(), 0)


class ModelGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_initial_model_matches_r2_and_disabled_branch_has_no_weights(self):
        baseline = make_model(**R2_OPTIONS).eval()
        enhanced = make_model(**NEW_OPTIONS).eval()
        result = enhanced.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(k.startswith('dsm_geometry_branch.') for k in result.missing_keys))
        self.assertFalse(any(k.startswith('dsm_geometry_branch.') for k in baseline.state_dict()))
        rgb, dsm = torch.rand(1, 3, 64, 64), torch.rand(1, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(enhanced(rgb, dsm), baseline(rgb, dsm), atol=0, rtol=0)
        restored = make_model(**dict(R2_OPTIONS, use_dsm_geometry_branch=False))
        restored.load_state_dict(baseline.state_dict(), strict=True)

    def test_aux_supervision_reaches_branch_and_corrected_features_reach_heads(self):
        model = make_model(**NEW_OPTIONS).train()
        # Check wiring after activating the residual, beyond the initial identity.
        for injection in model.dsm_geometry_branch.injections:
            nn.init.normal_(injection.projection.weight, std=.01)
        seen = {}
        handles = [model.dsm_geometry_branch.register_forward_hook(
            lambda module, inputs, output: seen.update(corrected=output))]
        for i, head in enumerate(model.dsm_category_heads):
            handles.append(head.register_forward_pre_hook(
                lambda module, inputs, i=i: seen.update({i: inputs[0]})))
        rgb, dsm = torch.rand(2, 3, 64, 64), torch.rand(2, 64, 64)
        output = model(rgb, dsm)
        for i in range(4):
            self.assertIs(seen[i], seen['corrected'][i])
        target = torch.randint(0, 6, (2, 64, 64))
        _, terms = segmentation_loss(output, target, category_weight=.1, pan_weight=.05)
        terms['dsm'].backward()  # Auxiliary DSM supervision alone must reach geometry.
        self.assertGreater(model.dsm_geometry_branch.encoder.stem[0].weight.grad.abs().sum().item(), 0)
        for injection in model.dsm_geometry_branch.injections:
            self.assertGreater(injection.gate[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for n,p in model.image_encoder.named_parameters()
                            if 'lora_' not in n))
        for handle in handles:
            handle.remove()
        model.eval()
        with torch.no_grad():
            expected = model(rgb, dsm)
        checkpoint = dict(model_state_dict=model.state_dict(), model_options=NEW_OPTIONS, num_classes=6)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        restored = make_model(**NEW_OPTIONS).eval()
        restored.load_state_dict(checkpoint_state_dict(torch.load(buffer, weights_only=True),
                                                       NEW_OPTIONS, 6), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(rgb, dsm), expected)

    def test_geometry_can_be_ablated_independently_of_first_contribution(self):
        options = dict(use_category_supervision=False, use_confidence_fusion=False,
                       use_category_pan=False, use_dsm_geometry=False,
                       use_dsm_geometry_branch=True)
        model = make_model(**options).eval()
        with torch.no_grad():
            result = model(torch.rand(1, 3, 64, 64), torch.rand(1, 64, 64))
        self.assertEqual(result.shape, (1, 6, 64, 64))
        with self.assertRaises(ValueError):
            make_model(use_dsm_geometry=True, use_dsm_geometry_branch=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_forward_backward(self):
        model = make_model(**NEW_OPTIONS).cuda().train()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
        rgb, dsm = torch.rand(2, 3, 64, 64, device='cuda'), torch.rand(2, 64, 64, device='cuda')
        target = torch.randint(0, 6, (2, 64, 64), device='cuda')
        for _ in range(2):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, _ = segmentation_loss(model(rgb, dsm), target, category_weight=.1, pan_weight=.05)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_old_r2_metadata_defaults_to_disabled_new_branch(self):
        weights = {'example': torch.tensor(1)}
        checkpoint = dict(model_state_dict=weights, model_options=R2_OPTIONS, num_classes=6)
        options = dict(R2_OPTIONS, use_dsm_geometry_branch=False, dsm_geometry_windows=(17,45))
        self.assertIs(checkpoint_state_dict(checkpoint, options, 6), weights)
        with self.assertRaises(ValueError):
            checkpoint_state_dict(checkpoint, NEW_OPTIONS, 6)

    def test_new_checkpoint_rejects_wrong_active_geometry_and_accepts_list_windows(self):
        checkpoint = dict(model_state_dict={}, model_options=NEW_OPTIONS, num_classes=6)
        checkpoint_state_dict(checkpoint, dict(NEW_OPTIONS, dsm_geometry_windows=[9,25]), 6)
        for options, classes in [(R2_OPTIONS,6), (dict(NEW_OPTIONS,dsm_geometry_windows=(17,45)),6),
                                 (NEW_OPTIONS,5)]:
            with self.assertRaises(ValueError):
                checkpoint_state_dict(checkpoint, options, classes)

    def test_plain_legacy_state_only_accepts_no_added_modules(self):
        options = dict(use_category_supervision=False, use_confidence_fusion=False,
                       use_category_pan=False, use_dsm_geometry=False,
                       use_n2_category_gate=False, use_dsm_geometry_branch=False,
                       dsm_geometry_windows=(9,25))
        weights = {'example': torch.tensor(1)}
        self.assertIs(checkpoint_state_dict(weights, options, 6), weights)
        with self.assertRaises(ValueError):
            checkpoint_state_dict(weights, dict(options, use_dsm_geometry_branch=True), 6)


if __name__ == '__main__':
    unittest.main()
