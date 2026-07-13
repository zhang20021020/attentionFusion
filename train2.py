import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from DoubleSwinMambnClean import UNetFormer_TwoModal
from utils2 import (
    DATASET,
    DATA_FOLDER,
    DSM_FOLDER,
    LABEL_FOLDER,
    N_CLASSES,
    VIS_CAM_SCALE,
    VIS_IDS,
    VIS_MAX_PATCHES,
    VIS_OUT_ROOT,
    VIS_PATCH_SIZE,
    VIS_WEIGHT_PATH,
    convert_from_color,
    convert_to_color,
    io,
    set_seed,
    sliding_window,
    test_ids,
)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
    return checkpoint


def load_opt002_weights(model, weight_path, device):
    if not weight_path:
        raise ValueError(
            "VIS_WEIGHT_PATH is empty. Set it to a mamba_opt002 checkpoint, for example: "
            "VIS_WEIGHT_PATH=./resultsp/UNetformer_epochXX_XXXX.pth python train2.py"
        )
    if not os.path.isfile(weight_path):
        raise FileNotFoundError("Visualization checkpoint not found: {}".format(weight_path))

    state_dict = extract_state_dict(torch.load(weight_path, map_location=device))
    model_state = model.state_dict()
    mismatched = [
        "{}: checkpoint {} != model {}".format(
            key, tuple(value.shape), tuple(model_state[key].shape)
        )
        for key, value in state_dict.items()
        if key in model_state and value.shape != model_state[key].shape
    ]
    if mismatched:
        preview = "\n".join(mismatched[:10])
        raise RuntimeError(
            "Checkpoint is not compatible with mamba_opt002. Shape mismatches:\n{}".format(preview)
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is not an exact mamba_opt002 match.\nMissing keys: {}\nUnexpected keys: {}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )


def normalize_cam(cam):
    cam = F.relu(cam)
    flat = cam.flatten(1)
    cam_min = flat.min(dim=1).values[:, None, None]
    cam_max = flat.max(dim=1).values[:, None, None]
    return (cam - cam_min) / (cam_max - cam_min + 1e-6)


def gradcam_from_feature(feature, gradient, out_size):
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = normalize_cam((weights * feature).sum(dim=1))
    return F.interpolate(
        cam.unsqueeze(1),
        size=out_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def branch_modules(model, scale):
    if scale not in {"low", "mid", "high"}:
        raise ValueError("VIS_CAM_SCALE must be one of: low, mid, high.")
    return (
        getattr(model.encoder, "to256_rgb_{}".format(scale)),
        getattr(model.encoder, "to256_dsm_{}".format(scale)),
    )


def predict_with_branch_gradcams(model, rgb_tensor, dsm_tensor, scale):
    features = {}

    def capture(name):
        def hook(_module, _inputs, output):
            features[name] = output
        return hook

    rgb_module, dsm_module = branch_modules(model, scale)
    handles = [
        rgb_module.register_forward_hook(capture("rgb")),
        dsm_module.register_forward_hook(capture("dsm")),
    ]
    try:
        model.zero_grad(set_to_none=True)
        logits = model(rgb_tensor, dsm_tensor, mode="VIS")
        predicted_classes = logits.argmax(dim=1, keepdim=True)
        score = logits.gather(1, predicted_classes).mean()
        rgb_grad, dsm_grad = torch.autograd.grad(
            score,
            (features["rgb"], features["dsm"]),
            retain_graph=False,
            create_graph=False,
        )
        out_size = rgb_tensor.shape[-2:]
        rgb_cam = gradcam_from_feature(features["rgb"], rgb_grad, out_size)
        dsm_cam = gradcam_from_feature(features["dsm"], dsm_grad, out_size)
        return logits.detach(), rgb_cam.detach(), dsm_cam.detach()
    finally:
        for handle in handles:
            handle.remove()


def read_tile(tile_id):
    rgb = io.imread(DATA_FOLDER.format(tile_id))
    if DATASET == "Potsdam":
        rgb = rgb[:, :, :3]
    rgb = rgb.astype("float32") / 255.0

    dsm = io.imread(DSM_FOLDER.format(tile_id)).astype("float32")
    dsm = (dsm - dsm.min()) / (dsm.max() - dsm.min() + 1e-6)
    gt = convert_from_color(io.imread(LABEL_FOLDER.format(tile_id))).astype("int64")
    return rgb, dsm, gt


def save_patch_images(out_dir, prefix, rgb_patch, dsm_patch, gt_patch, pred_patch, rgb_cam, dsm_cam):
    rgb = np.clip(rgb_patch * 255.0, 0, 255).astype(np.uint8)
    dsm = np.clip(dsm_patch * 255.0, 0, 255).astype(np.uint8)
    gt = convert_to_color(gt_patch)
    pred = convert_to_color(pred_patch)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    dsm_bgr = cv2.cvtColor(dsm, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    rgb_heat = cv2.applyColorMap((rgb_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    dsm_heat = cv2.applyColorMap((dsm_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb_overlay = cv2.addWeighted(rgb_bgr, 0.5, rgb_heat, 0.5, 0)
    dsm_overlay = cv2.addWeighted(dsm_bgr, 0.5, dsm_heat, 0.5, 0)
    comparison = cv2.hconcat([rgb_bgr, dsm_bgr, gt_bgr, pred_bgr, rgb_overlay, dsm_overlay])

    Image.fromarray(rgb).save(os.path.join(out_dir, prefix + "_rgb.png"))
    Image.fromarray(dsm).save(os.path.join(out_dir, prefix + "_dsm.png"))
    Image.fromarray(gt).save(os.path.join(out_dir, prefix + "_gt.png"))
    Image.fromarray(pred).save(os.path.join(out_dir, prefix + "_pred.png"))
    cv2.imwrite(os.path.join(out_dir, prefix + "_rgb_heat.png"), rgb_heat)
    cv2.imwrite(os.path.join(out_dir, prefix + "_dsm_heat.png"), dsm_heat)
    cv2.imwrite(os.path.join(out_dir, prefix + "_rgb_heat_overlay.png"), rgb_overlay)
    cv2.imwrite(os.path.join(out_dir, prefix + "_dsm_heat_overlay.png"), dsm_overlay)
    cv2.imwrite(os.path.join(out_dir, prefix + "_comparison.png"), comparison)


def export_visualizations(model, ids, device, patch_size, cam_scale, out_root, max_patches=0):
    model.eval()
    out_dir = os.path.join(out_root, "{}_test_{}".format(DATASET, cam_scale))
    os.makedirs(out_dir, exist_ok=True)
    print("[*] Output directory: {}".format(out_dir))

    exported = 0
    for tile_id in ids:
        print("[*] Processing tile: {}".format(tile_id))
        rgb_full, dsm_full, gt_full = read_tile(tile_id)
        patch_index = 0

        for x, y, w, h in sliding_window(
                rgb_full, step=patch_size, window_size=(patch_size, patch_size)):
            rgb_patch = rgb_full[x:x + w, y:y + h]
            dsm_patch = dsm_full[x:x + w, y:y + h]
            gt_patch = gt_full[x:x + w, y:y + h]

            rgb_tensor = torch.from_numpy(
                rgb_patch.transpose(2, 0, 1)
            ).unsqueeze(0).float().to(device)
            dsm_tensor = torch.from_numpy(
                dsm_patch
            ).unsqueeze(0).unsqueeze(0).float().to(device)

            logits, rgb_cam, dsm_cam = predict_with_branch_gradcams(
                model, rgb_tensor, dsm_tensor, cam_scale
            )
            pred_patch = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            rgb_cam = rgb_cam.squeeze(0).cpu().numpy()
            dsm_cam = dsm_cam.squeeze(0).cpu().numpy()

            patch_index += 1
            exported += 1
            prefix = "{}_x{:04d}_y{:04d}_p{:03d}".format(
                tile_id, x, y, patch_index
            )
            save_patch_images(
                out_dir, prefix, rgb_patch, dsm_patch, gt_patch,
                pred_patch, rgb_cam, dsm_cam
            )

            if max_patches > 0 and exported >= max_patches:
                print("[*] Reached VIS_MAX_PATCHES={}".format(max_patches))
                return


def main():
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mamba_opt002 visualization.")

    device = torch.device("cuda:0")
    model = UNetFormer_TwoModal(num_classes=N_CLASSES).to(device)
    print("[*] Loading mamba_opt002 weights: {}".format(VIS_WEIGHT_PATH))
    load_opt002_weights(model, VIS_WEIGHT_PATH, device)

    ids = VIS_IDS or test_ids
    export_visualizations(
        model=model,
        ids=ids,
        device=device,
        patch_size=VIS_PATCH_SIZE,
        cam_scale=VIS_CAM_SCALE,
        out_root=VIS_OUT_ROOT,
        max_patches=VIS_MAX_PATCHES,
    )


if __name__ == "__main__":
    main()
