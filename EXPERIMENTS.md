# Category-guided fusion experiments

> Current extension: the five-channel DSM residual branch is documented in
> [DSM_GEOMETRY.md](DSM_GEOMETRY.md). The current utils.py uses the R2 settings
> (legacy Sobel input off, N2 gate off, category/pan weights 0.1/0.05) plus the
> new `use_dsm_geometry_branch=True`. The configuration and local verification
> below describe the earlier category-guidance experiments, not these defaults.

This is an adaptation of the category supervision/confidence ideas in
CSFAFormer to the existing SAM + LoRA, SE fusion, BAM, PAN and CIAPNet head.
It is not a full reproduction of CSFAFormer or its CSTM attention module.

## Configuration and training

Edit `utils.py`. The default is `MODE = 'Train'` with all four options enabled:

```python
MODEL_OPTIONS = dict(
    use_category_supervision=True,
    use_confidence_fusion=True,
    use_category_pan=True,
    use_dsm_geometry=True,
)
LOSS_OPTIONS = dict(category_weight=0.4, pan_weight=0.2,
                    decoder_weight=0.0, ignore_index=6)
```

Confirm the dataset paths in `utils.py` and the pretrained encoder file
`weights/sam_vit_l_0b3195.pth`, then run in your training environment:

```bash
python train.py
```

Training prints the experiment flags and each unweighted loss component.
The existing optimizer, learning rate, splits, crop size and epoch budget are
retained. The first scheduled validation can now save a checkpoint even if
mIoU is below the previous hard-coded threshold of 0.83.

## Implemented computation

1. DSM is normalized per full tile before cropping in both training and testing.
   Constant tiles become zero; nonfinite values are replaced by the finite
   minimum. All-invalid tiles become zero.
2. The model constructs `[DSM, Sobel_x(DSM), Sobel_y(DSM)]` after augmentation.
   Sobel uses replicate padding and fixed kernels divided by 8. Gradients stay
   signed, approximately in [-0.5, 0.5] for a [0, 1] DSM. There is no per-patch
   gradient normalization that could amplify flat-region noise. Disabling the
   option restores the repeated three-channel DSM input.
3. Separate optical/DSM category heads predict six-class logits at all four
   pyramid scales. Their logits are supervised with ground truth only during
   training. Inference needs only the two input images.
4. Confidence is `(1 - normalized_entropy) * (top1 - top2)`. For each class,
   each modality's support is `confidence * class_probability`. Normalized
   support gives the optical modality share, with an equal-share fallback.
5. At the first three scales, learned channel-to-class associations convert
   class-wise shares into channel/spatial fusion weights. The result is added
   residually to the existing SE sum, with a bounded learned strength initially
   equal to sigmoid(-2), about 0.119. The fourth scale retains plain SE fusion.
6. Calibrated category distributions propagate from coarse to fine through
   learned mixing weights. They gate N2 and the bottom-up transfers into
   N3/N4/N5, with confidence-scaled residual modulation. Local PAN additions,
   BAM and the original decoder remain in the feature path. Separate PAN heads
   supervise the four resulting feature maps.

Confidence and modality probabilities are detached before routing. This keeps
the routing objective from encouraging artificial certainty; the explicit
category losses train the prediction heads and LoRA/FPN. The PAN mixing weights,
PAN gates and channel-to-class associations receive segmentation gradients.

## Loss and ablations

Each auxiliary family is averaged across its four scales:

```text
L = CE(main)
  + 0.4 * (mean CE(optical_heads) + mean CE(DSM_heads))
  + 0.2 * mean CE(PAN_heads)
  + 0.0 * CE(existing_decoder_aux)
```

Logits are bilinearly resized to the target size before CE. Label 6 is ignored;
clutter remains class 5. An all-ignored crop produces a finite zero loss.
The original decoder auxiliary loss is opt-in so it does not confound the
requested experiments. Fusion or category PAN requires category supervision
and a positive category loss weight in the training script.

| Experiment | Supervision | Confidence fusion | Category PAN | DSM geometry |
| --- | --- | --- | --- | --- |
| Original architecture | False | False | False | False |
| Auxiliary supervision only | True | False | False | False |
| Add confidence fusion | True | True | False | False |
| Add category PAN | True | True | True | False |
| Complete model | True | True | True | True |
| Geometry alone | False | False | False | True |

Keep `decoder_weight=0` across the primary comparisons. To isolate PAN routing
from its extra supervision, also compare category PAN with `pan_weight=0`.
Use the same seeds and training budget; compare per-class IoU/F1 and boundaries
in addition to aggregate mIoU. These changes do not establish an accuracy gain.

## Checkpoints and inference

New `.pth` checkpoints contain `model_state_dict`, architecture/loss options,
class count, epoch, score, seed, dataset split and crop size. Their names include
the four architecture flags. They are saved under `resultsv` or `resultsp`.

Set `MODE = 'Test'`, set `CHECKPOINT_PATH` to the trained `.pth`, and keep
`MODEL_OPTIONS` identical to its saved options. Run `python train.py`.
Loading is strict to avoid silently testing randomly initialized new modules.
Legacy plain state dictionaries are accepted only with all options disabled.
Other scripts that load a new checkpoint must extract `model_state_dict` and
instantiate `UNetFormer` with its saved `model_options`.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover geometry, normalization, confidence fallbacks, supervised gradients,
PAN propagation, ablation combinations, save/load equivalence and CUDA AMP
when available. Integration tests use the real SAM/LoRA encoder implementation
at reduced width/depth without pretrained weights, and the production decoder.
They do not replace full-resolution pretrained training or held-out evaluation.

Local verification: all 13 tests passed, including a CUDA forward/backward pass
at the configured 256x256 crop and window size 8. An additional comparison with
the pre-change Git version confirmed strict state compatibility and bit-exact
evaluation outputs when all four options are disabled (shared test weights).
The configured pretrained checkpoint and Vaihingen dataset directory were not
present locally, so no dataset accuracy measurement was performed.
