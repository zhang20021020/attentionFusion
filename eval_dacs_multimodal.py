import argparse
import os

import numpy as np
import torch
from skimage import io

from DoubleSwinMambnClean import UNetFormer_TwoModal as MFNet
from train_dacs_multimodal import DOMAIN_IDS, LABELS, N_CLASSES, convert_from_color, domain_paths


def label_path(data_root, domain, tile_id, use_eroded=True):
    root = os.path.join(data_root, domain)
    if domain == "Vaihingen":
        eroded = os.path.join(
            root,
            "gts_eroded_for_participants",
            f"top_mosaic_09cm_area{tile_id}_noBoundary.tif",
        )
        raw = os.path.join(
            root,
            "gts_for_participants",
            f"top_mosaic_09cm_area{tile_id}.tif",
        )
    elif domain in ("Potsdam", "Potsdam2"):
        eroded = os.path.join(
            root,
            "5_Labels_for_participants_no_Boundary",
            f"top_potsdam_{tile_id}_label_noBoundary.tif",
        )
        raw = os.path.join(
            root,
            "5_Labels_for_participants",
            f"top_potsdam_{tile_id}_label.tif",
        )
    else:
        raise ValueError(f"Unknown domain: {domain}")

    if use_eroded and os.path.isfile(eroded):
        return eroded
    return raw


def read_tile(data_root, domain, tile_id, use_eroded=True):
    paths = domain_paths(data_root, domain)
    rgb = io.imread(paths["rgb"].format(tile_id))[:, :, :3]
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0

    dsm = np.asarray(io.imread(paths["dsm"].format(tile_id)), dtype=np.float32)
    dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-6)

    label = convert_from_color(io.imread(label_path(data_root, domain, tile_id, use_eroded)))
    return rgb, dsm, label


def sliding_window(height, width, crop_size, stride):
    crop_h, crop_w = crop_size
    for x in range(0, height, stride):
        if x + crop_h > height:
            x = height - crop_h
        for y in range(0, width, stride):
            if y + crop_w > width:
                y = width - crop_w
            yield x, y, crop_h, crop_w


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@torch.no_grad()
def predict_tile(model, rgb, dsm, device, crop_size, stride, batch_size):
    model.eval()
    height, width = rgb.shape[:2]
    logits_sum = np.zeros((N_CLASSES, height, width), dtype=np.float32)
    count = np.zeros((1, height, width), dtype=np.float32)

    windows = sliding_window(height, width, crop_size, stride)
    for coords in batched(windows, batch_size):
        rgb_patches = []
        dsm_patches = []
        for x, y, h, w in coords:
            rgb_patch = rgb[x:x + h, y:y + w].transpose(2, 0, 1)
            dsm_patch = dsm[x:x + h, y:y + w]
            rgb_patches.append(rgb_patch)
            dsm_patches.append(dsm_patch)

        rgb_tensor = torch.from_numpy(np.stack(rgb_patches)).float().to(device)
        dsm_tensor = torch.from_numpy(np.stack(dsm_patches)).float().to(device)
        logits = model(rgb_tensor, dsm_tensor, mode="Test").detach().cpu().numpy()

        for out, (x, y, h, w) in zip(logits, coords):
            logits_sum[:, x:x + h, y:y + w] += out
            count[:, x:x + h, y:y + w] += 1

    logits_sum /= np.maximum(count, 1.0)
    return np.argmax(logits_sum, axis=0).astype(np.uint8)


def confusion_matrix(pred, target, num_classes):
    valid = (target >= 0) & (target < num_classes)
    indices = num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_metrics(cm):
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    union = gt + pred - tp

    iou = tp / np.maximum(union, 1.0)
    f1 = 2.0 * tp / np.maximum(gt + pred, 1.0)
    acc = tp / np.maximum(gt, 1.0)
    oa = tp.sum() / np.maximum(cm.sum(), 1.0)

    return {
        "OA": oa,
        "mIoU_all": np.nanmean(iou),
        "mIoU_no_clutter": np.nanmean(iou[:-1]),
        "mF1_all": np.nanmean(f1),
        "mF1_no_clutter": np.nanmean(f1[:-1]),
        "IoU": iou,
        "F1": f1,
        "Acc": acc,
    }


def load_weights(model, checkpoint_path, weight_key, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"DACS checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and weight_key in checkpoint:
        state_dict = checkpoint[weight_key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in: {checkpoint_path}")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {weight_key} weights from: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(
            "The evaluation checkpoint does not exactly match the current model; "
            "refusing to report potentially invalid Experiment C metrics."
        )


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MFNet(
        num_classes=N_CLASSES,
        pretrained=False,
        weight_path=None,
    ).to(device)
    load_weights(model, args.checkpoint, args.weight_key, device)

    ids = args.ids.split(",") if args.ids else DOMAIN_IDS[args.domain][args.split]
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    for tile_id in ids:
        rgb, dsm, label = read_tile(args.data_root, args.domain, tile_id, use_eroded=not args.raw_label)
        pred = predict_tile(
            model,
            rgb,
            dsm,
            device,
            crop_size=(args.crop_size, args.crop_size),
            stride=args.stride,
            batch_size=args.batch_size,
        )
        cm += confusion_matrix(pred, label, N_CLASSES)
        print(f"Evaluated tile {tile_id}")

    metrics = compute_metrics(cm)
    print("\nConfusion matrix:")
    print(cm)
    print(f"\nOA                 : {metrics['OA']:.4f}")
    print(f"mIoU (all 6)       : {metrics['mIoU_all']:.4f}")
    print(f"mIoU (no clutter)  : {metrics['mIoU_no_clutter']:.4f}")
    print(f"mF1  (all 6)       : {metrics['mF1_all']:.4f}")
    print(f"mF1  (no clutter)  : {metrics['mF1_no_clutter']:.4f}")
    print("\nPer-class metrics:")
    for idx, name in enumerate(LABELS):
        print(
            f"{idx} {name:10s} "
            f"IoU={metrics['IoU'][idx]:.4f} "
            f"F1={metrics['F1'][idx]:.4f} "
            f"Acc={metrics['Acc'][idx]:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DACS multimodal checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/home/zhangben/ISPRS_dataset/")
    parser.add_argument("--domain", choices=sorted(DOMAIN_IDS), default="Vaihingen")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--ids", default="", help="Comma-separated tile ids. Overrides --split.")
    parser.add_argument("--weight-key", choices=["model", "teacher"], default="teacher")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--raw-label", action="store_true", help="Use raw labels instead of noBoundary labels.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
