# [arxiv] MFNet

This repo is the official implementation of ['MFNet: Fine-Tuning Segment Anything Model for Multimodal Remote Sensing Semantic Segmentation'](https://arxiv.org/abs/2410.11160).

![framework](https://github.com/sstary/SSRS/blob/main/docs/MFNet.png)

## Usage
You can get the pre-trained model here: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

The core modules are in ./MedSAM/models/ImageEncoder and ./MedSAM/models/sam

The current mode is MMLoRA. You can choose MMAdapter or MMLoRA by the 'mod' hyper-parameter in **./MedSAM/cfg.py**, and also need modify the Line 523/524 in **SSRS/MFNet
/UNetFormer_MMSAM.py** and Line 35/36, 42/43 in **SSRS/MFNet/train.py**. Most of the remaining files come from the original SAM framework and can be ignored.


Run the code by: python train.py

Draw the heatmap by: python test_heatmap.py

## Experiment C: Potsdam2 9 cm -> DACS -> Vaihingen

Experiment C starts from the source-only checkpoint produced by Experiment B,
uses labeled `Potsdam2` patches as the source domain, and uses unlabeled
`Vaihingen` training patches as the DACS target domain. The default checkpoint
matches the Experiment B path currently used by `train.py`.

```bash
python train_dacs_multimodal.py \
  --source Potsdam2 \
  --target Vaihingen \
  --source-checkpoint ./resultsp2/UNetformer_epoch47_0.8442.pth \
  --out-dir ./results_experiment_c
```

Only after adaptation is complete, evaluate the EMA teacher on the held-out
Vaihingen test tiles:

```bash
python eval_dacs_multimodal.py \
  --checkpoint ./results_experiment_c/dacs_multimodal_iter_4000.pth \
  --weight-key teacher \
  --domain Vaihingen \
  --split test
```

The adaptation dataset never loads Vaihingen labels. The evaluation command is
kept separate so the test labels do not influence DACS training or checkpoint
selection.

Please cite our paper if you find it is useful for your research.

```
@article{ma2024manet,
  title={MANet: Fine-Tuning Segment Anything Model for Multimodal Remote Sensing Semantic Segmentation},
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On and Huang, Bo},
  journal={arXiv preprint arXiv:2410.11160},
  year={2024}
}
  ```
