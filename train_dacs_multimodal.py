import argparse
import copy
import os
import random
import time
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torch.utils.data import DataLoader, Dataset

from DoubleSwinMambnClean import UNetFormer_TwoModal as MFNet


LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
N_CLASSES = len(LABELS)
IGNORE_INDEX = 255
DEFAULT_SOURCE_CHECKPOINT = "./resultsp2/UNetformer_epoch47_0.8442.pth"

PALETTE = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    6: (0, 0, 0),
}
INVERT_PALETTE = {v: k for k, v in PALETTE.items()}


POTSDAM_IDS = {
    "train": [
        "6_10", "7_10", "2_12", "3_11", "2_10", "7_8", "5_10", "3_12",
        "5_12", "7_11", "7_9", "6_9", "7_7", "4_12", "6_8", "6_12",
        "6_7", "4_11",
    ],
    "test": ["4_10", "5_11", "2_11", "3_10", "6_11", "7_12"],
}


DOMAIN_IDS = {
    "Vaihingen": {
        "train": ["1", "3", "23", "26", "7", "11", "13", "28", "17", "32", "34", "37"],
        "test": ["5", "21", "15", "30"],
    },
    "Potsdam": POTSDAM_IDS,
    # Potsdam2 contains the same tiles downsampled from 5 cm to 9 cm GSD.
    "Potsdam2": POTSDAM_IDS,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_from_color(arr_3d):
    arr_2d = np.full(arr_3d.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for color, class_id in INVERT_PALETTE.items():
        mask = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
        arr_2d[mask] = class_id if class_id < N_CLASSES else IGNORE_INDEX
    return arr_2d


def domain_paths(data_root, domain):
    root = os.path.join(data_root, domain)
    if domain == "Vaihingen":
        return {
            "rgb": os.path.join(root, "top", "top_mosaic_09cm_area{}.tif"),
            "dsm": os.path.join(root, "dsm", "dsm_09cm_matching_area{}.tif"),
            "label": os.path.join(root, "gts_for_participants", "top_mosaic_09cm_area{}.tif"),
        }
    if domain in ("Potsdam", "Potsdam2"):
        return {
            "rgb": os.path.join(root, "4_Ortho_RGBIR", "top_potsdam_{}_RGBIR.tif"),
            "dsm": os.path.join(root, "1_DSM_normalisation", "dsm_potsdam_{}_normalized_lastools.jpg"),
            "label": os.path.join(root, "5_Labels_for_participants", "top_potsdam_{}_label.tif"),
        }
    raise ValueError(f"Unknown domain: {domain}")


def random_crop_pos(shape, crop_size):
    h, w = shape
    crop_h, crop_w = crop_size
    if h < crop_h or w < crop_w:
        raise ValueError(f"Image shape {(h, w)} is smaller than crop size {crop_size}")
    x1 = random.randint(0, h - crop_h)
    y1 = random.randint(0, w - crop_w)
    return x1, x1 + crop_h, y1, y1 + crop_w


def augment(rgb, dsm, label=None):
    if random.random() < 0.5:
        rgb = rgb[:, ::-1, :]
        dsm = dsm[::-1, :]
        if label is not None:
            label = label[::-1, :]
    if random.random() < 0.5:
        rgb = rgb[:, :, ::-1]
        dsm = dsm[:, ::-1]
        if label is not None:
            label = label[:, ::-1]
    return rgb.copy(), dsm.copy(), None if label is None else label.copy()


class DomainPatchDataset(Dataset):
    def __init__(self, domain, ids, data_root, crop_size, require_label=True, cache=False, length=10000):
        self.domain = domain
        self.ids = ids
        self.paths = domain_paths(data_root, domain)
        self.crop_size = crop_size
        self.require_label = require_label
        self.cache = cache
        self.length = length
        self.rgb_cache = {}
        self.dsm_cache = {}
        self.label_cache = {}

        for tile_id in ids:
            required = [self.paths["rgb"].format(tile_id), self.paths["dsm"].format(tile_id)]
            if require_label:
                required.append(self.paths["label"].format(tile_id))
            for path in required:
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)

    def __len__(self):
        return self.length

    def _read_rgb(self, idx):
        if idx not in self.rgb_cache:
            arr = io.imread(self.paths["rgb"].format(self.ids[idx]))
            arr = arr[:, :, :3].transpose(2, 0, 1)
            arr = np.asarray(arr, dtype=np.float32) / 255.0
            if self.cache:
                self.rgb_cache[idx] = arr
            return arr
        return self.rgb_cache[idx]

    def _read_dsm(self, idx):
        if idx not in self.dsm_cache:
            arr = np.asarray(io.imread(self.paths["dsm"].format(self.ids[idx])), dtype=np.float32)
            arr_min, arr_max = np.min(arr), np.max(arr)
            arr = (arr - arr_min) / max(arr_max - arr_min, 1e-6)
            if self.cache:
                self.dsm_cache[idx] = arr
            return arr
        return self.dsm_cache[idx]

    def _read_label(self, idx):
        if idx not in self.label_cache:
            arr = convert_from_color(io.imread(self.paths["label"].format(self.ids[idx])))
            if self.cache:
                self.label_cache[idx] = arr
            return arr
        return self.label_cache[idx]

    def __getitem__(self, _):
        idx = random.randint(0, len(self.ids) - 1)
        rgb = self._read_rgb(idx)
        dsm = self._read_dsm(idx)
        label = self._read_label(idx) if self.require_label else None

        x1, x2, y1, y2 = random_crop_pos(rgb.shape[-2:], self.crop_size)
        rgb_patch = rgb[:, x1:x2, y1:y2]
        dsm_patch = dsm[x1:x2, y1:y2]
        label_patch = label[x1:x2, y1:y2] if label is not None else None
        rgb_patch, dsm_patch, label_patch = augment(rgb_patch, dsm_patch, label_patch)

        sample = {
            "rgb": torch.from_numpy(rgb_patch).float(),
            "dsm": torch.from_numpy(dsm_patch).float(),
        }
        if label_patch is not None:
            sample["label"] = torch.from_numpy(label_patch).long()
        return sample


def classmix_mask(labels, num_classes=N_CLASSES, ignore_index=IGNORE_INDEX):
    masks = []
    for label in labels:
        classes = torch.unique(label)
        classes = classes[(classes != ignore_index) & (classes < num_classes)]
        if classes.numel() == 0:
            masks.append(torch.zeros_like(label, dtype=torch.float32).unsqueeze(0))
            continue
        choice_count = int((classes.numel() + classes.numel() % 2) / 2)
        choice = classes[torch.randperm(classes.numel(), device=label.device)[:choice_count]]
        masks.append(torch.isin(label, choice).float().unsqueeze(0))
    return torch.stack(masks, dim=0)


def local_pseudo_weight(pseudo_prob, threshold, kernel_size):
    if kernel_size <= 0:
        valid = pseudo_prob.ge(threshold).float()
        ratio = valid.flatten(1).mean(dim=1).view(-1, 1, 1)
        return ratio.expand_as(pseudo_prob)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=pseudo_prob.device)
    valid = pseudo_prob.ge(threshold).float().unsqueeze(1)
    weight = F.conv2d(valid, kernel, padding=kernel_size // 2)
    return (weight / float(kernel_size * kernel_size)).squeeze(1)


def weighted_cross_entropy(logits, target, pixel_weight=None):
    loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX, reduction="none")
    valid = target.ne(IGNORE_INDEX).float()
    if pixel_weight is None:
        denom = valid.sum().clamp_min(1.0)
        return (loss * valid).sum() / denom
    weight = pixel_weight.float() * valid
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def update_ema(student, teacher, alpha):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.data.copy_(student_buffer.data)


def load_checkpoint(model, path, device, allow_partial=False):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Source checkpoint does not exist: {path}. "
            "Experiment C must start from the source-only Experiment B weights."
        )
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in: {path}")
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if (missing or unexpected) and not allow_partial:
        raise RuntimeError(
            "The Experiment B checkpoint does not exactly match the DACS model. "
            "Use --allow-partial-checkpoint only when this mismatch is intentional."
        )


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    crop_size = (args.crop_size, args.crop_size)
    amp_enabled = args.amp and device.type == "cuda"

    if args.source == args.target:
        raise ValueError("DACS requires different source and target domains.")
    if not args.source_checkpoint:
        raise ValueError(
            "--source-checkpoint is required for Experiment C; pass the Experiment B weights."
        )

    print("Experiment C: source-only 9 cm model -> DACS UDA -> Vaihingen")
    print(f"Source domain      : {args.source}")
    print(f"Target domain      : {args.target} (labels are not loaded)")
    print(f"Source checkpoint  : {args.source_checkpoint}")
    print(f"AMP enabled        : {amp_enabled}")

    source_dataset = DomainPatchDataset(
        args.source,
        DOMAIN_IDS[args.source]["train"],
        args.data_root,
        crop_size,
        require_label=True,
        cache=args.cache,
        length=args.source_length,
    )
    target_dataset = DomainPatchDataset(
        args.target,
        DOMAIN_IDS[args.target]["train"],
        args.data_root,
        crop_size,
        require_label=False,
        cache=args.cache,
        length=args.target_length,
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = MFNet(
        num_classes=N_CLASSES,
        pretrained=args.pretrained,
        weight_path=args.weight_path or None,
    ).to(device)
    if args.source_checkpoint:
        load_checkpoint(
            model,
            args.source_checkpoint,
            device,
            allow_partial=args.allow_partial_checkpoint,
        )

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)

    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    for iteration in range(1, args.iters + 1):
        source = next(source_iter)
        target = next(target_iter)
        rgb_s = source["rgb"].to(device, non_blocking=True)
        dsm_s = source["dsm"].to(device, non_blocking=True)
        label_s = source["label"].to(device, non_blocking=True)
        rgb_t = target["rgb"].to(device, non_blocking=True)
        dsm_t = target["dsm"].to(device, non_blocking=True)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits_s = model(rgb_s, dsm_s, mode="Train")
            loss_source = weighted_cross_entropy(logits_s, label_s)

        # Backpropagate the source loss first so its activation graph can be
        # released before the mixed forward pass. This keeps DACS practical on
        # a 24 GiB GPU without changing the accumulated gradient.
        scaler.scale(loss_source).backward()
        del logits_s

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            teacher.eval()
            logits_t = teacher(rgb_t, dsm_t, mode="Test")
            probs_t = torch.softmax(logits_t, dim=1)
            pseudo_prob, pseudo_label = probs_t.max(dim=1)
            pseudo_weight = local_pseudo_weight(
                pseudo_prob,
                threshold=args.pseudo_threshold,
                kernel_size=args.pseudo_kernel_size,
            )

        mask = classmix_mask(label_s)
        mixed_rgb = mask * rgb_s + (1.0 - mask) * rgb_t
        mixed_dsm = mask.squeeze(1) * dsm_s + (1.0 - mask.squeeze(1)) * dsm_t
        mixed_label = torch.where(mask.squeeze(1).bool(), label_s, pseudo_label)
        mixed_weight = mask.squeeze(1) + (1.0 - mask.squeeze(1)) * pseudo_weight

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits_mix = model(mixed_rgb, mixed_dsm, mode="Train")
            loss_mix = weighted_cross_entropy(logits_mix, mixed_label, mixed_weight)

        scaler.scale(args.mix_loss_weight * loss_mix).backward()
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, teacher, args.ema_alpha)

        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - start
            trusted = pseudo_weight.mean().item()
            loss_value = loss_source.item() + args.mix_loss_weight * loss_mix.item()
            print(
                f"iter {iteration:05d}/{args.iters} "
                f"loss={loss_value:.4f} src={loss_source.item():.4f} "
                f"mix={loss_mix.item():.4f} pseudo_w={trusted:.3f} "
                f"time={elapsed:.1f}s"
            )

        if iteration % args.save_interval == 0 or iteration == args.iters:
            ckpt_path = os.path.join(args.out_dir, f"dacs_multimodal_iter_{iteration}.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment C: initialize from the Potsdam2 9 cm source-only model, "
            "then run DACS RGB+DSM adaptation to Vaihingen."
        )
    )
    parser.add_argument("--data-root", default="/home/zhangben/ISPRS_dataset/")
    parser.add_argument("--source", choices=sorted(DOMAIN_IDS), default="Potsdam2")
    parser.add_argument("--target", choices=sorted(DOMAIN_IDS), default="Vaihingen")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--source-length", type=int, default=10000)
    parser.add_argument("--target-length", type=int, default=10000)
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-alpha", type=float, default=0.9)
    parser.add_argument("--pseudo-threshold", type=float, default=0.7)
    parser.add_argument("--pseudo-kernel-size", type=int, default=7)
    parser.add_argument("--mix-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--source-checkpoint",
        default=DEFAULT_SOURCE_CHECKPOINT,
        help="Experiment B source-only checkpoint used to initialize student and EMA teacher.",
    )
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Allow missing/unexpected model keys when loading Experiment B weights.",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--weight-path", default="")
    parser.add_argument("--out-dir", default="./results_experiment_c")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision (enabled by default).",
    )
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
