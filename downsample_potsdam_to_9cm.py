import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile


SCALE_5CM_TO_9CM = 5.0 / 9.0


FOLDERS = {
    "4_Ortho_RGBIR": {
        "patterns": ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"),
        "interpolation": cv2.INTER_AREA,
        "min_channels": 4,
    },
    "1_DSM_normalisation": {
        "patterns": ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"),
        "interpolation": cv2.INTER_AREA,
        "min_channels": None,
    },
    "5_Labels_for_participants": {
        "patterns": ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"),
        "interpolation": cv2.INTER_NEAREST,
        "min_channels": 3,
    },
    "5_Labels_for_participants_no_Boundary": {
        "patterns": ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"),
        "interpolation": cv2.INTER_NEAREST,
        "min_channels": 3,
    },
}


def collect_files(folder, patterns):
    files = []
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    return sorted(files)


def is_tiff(path):
    return path.suffix.lower() in (".tif", ".tiff")


def read_image(path):
    if is_tiff(path):
        image = tifffile.imread(path)

        # Some multi-band TIFF files use planar (C, H, W) storage.  Normalize
        # them to the channel-last layout expected by the training pipeline.
        if (
            image.ndim == 3
            and image.shape[0] in (3, 4)
            and image.shape[-1] not in (3, 4)
        ):
            image = np.moveaxis(image, 0, -1)
        return image

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_image(path, image):
    if is_tiff(path):
        photometric = (
            "rgb"
            if image.ndim == 3 and image.shape[-1] in (3, 4)
            else "minisblack"
        )
        tifffile.imwrite(
            path,
            image,
            compression=None,
            photometric=photometric,
            metadata=None,
        )
        return

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def resize_image(
    src_path,
    dst_path,
    scale,
    interpolation,
    min_channels=None,
    overwrite=False,
):
    if dst_path.exists() and not overwrite:
        return "skip"

    image = read_image(src_path)
    if min_channels is not None and (
        image.ndim != 3 or image.shape[-1] < min_channels
    ):
        raise RuntimeError(
            f"Expected at least {min_channels} channels in {src_path}, "
            f"but got shape {image.shape}."
        )

    height, width = image.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(dst_path, resized)
    return "write"


def downsample_potsdam(src_root, dst_root, scale, overwrite=False, dry_run=False):
    src_root = Path(src_root).expanduser().resolve()
    dst_root = Path(dst_root).expanduser().resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")
    if src_root == dst_root:
        raise ValueError(
            "Source and target directories must be different; refusing to "
            "overwrite the original 5cm Potsdam dataset."
        )

    total_written = 0
    total_skipped = 0

    print(f"Source: {src_root}")
    print(f"Target: {dst_root}")
    print(f"Scale : {scale:.6f}")
    print(f"Overwrite existing target files: {overwrite}")
    print("TIFF compression: NONE")

    for folder_name, cfg in FOLDERS.items():
        src_folder = src_root / folder_name
        dst_folder = dst_root / folder_name

        if not src_folder.is_dir():
            print(f"[WARN] Missing folder, skip: {src_folder}")
            continue

        files = collect_files(src_folder, cfg["patterns"])
        print(f"\n[{folder_name}] {len(files)} files")

        for src_path in files:
            rel_path = src_path.relative_to(src_folder)
            dst_path = dst_folder / rel_path

            if dry_run:
                print(f"  {src_path.name} -> {dst_path}")
                continue

            status = resize_image(
                src_path,
                dst_path,
                scale=scale,
                interpolation=cfg["interpolation"],
                min_channels=cfg["min_channels"],
                overwrite=overwrite,
            )
            if status == "write":
                total_written += 1
            else:
                total_skipped += 1

        if dry_run:
            continue

        print(f"  done -> {dst_folder}")

    if not dry_run:
        print(f"\nFinished. written={total_written}, skipped={total_skipped}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downsample ISPRS Potsdam from 5cm GSD to 9cm GSD."
    )
    parser.add_argument(
        "--src-root",
        default="/home/zhangben/ISPRS_dataset/Potsdam",
        help="Original Potsdam directory.",
    )
    parser.add_argument(
        "--dst-root",
        default="/home/zhangben/ISPRS_dataset/Potsdam2",
        help="Output directory for downsampled Potsdam.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=SCALE_5CM_TO_9CM,
        help="Resize scale. 5cm to 9cm is 5/9.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing files in the target directory.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Keep existing files in the target directory instead of replacing them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned files without writing outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    downsample_potsdam(
        src_root=args.src_root,
        dst_root=args.dst_root,
        scale=args.scale,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
