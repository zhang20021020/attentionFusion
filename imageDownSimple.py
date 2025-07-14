#!/usr/bin/env python3
import os
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from affine import Affine
from PIL import Image


def resample_raster(src_path: Path, dst_path: Path, target_res: float, resampling_method, orig_res: float = 0.05):
    """
    Resample a raster file (GeoTIFF) to the specified target resolution.
    Handles both georeferenced and non-georeferenced files.
    """
    with rasterio.open(src_path) as src:
        profile = src.meta.copy()

        if src.crs is None:
            # No CRS: treat as simple array scaling
            scale = orig_res / target_res
            new_width = int(src.width * scale)
            new_height = int(src.height * scale)
            transform = src.transform * Affine(scale, 0, 0, 0, scale, 0)
            profile.update({
                'transform': transform,
                'width': new_width,
                'height': new_height
            })

            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=resampling_method
            )
            os.makedirs(dst_path.parent, exist_ok=True)
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(data)
        else:
            # Georeferenced: use calculate_default_transform
            transform, width, height = calculate_default_transform(
                src.crs, src.crs,
                src.width, src.height,
                *src.bounds,
                resolution=(target_res, target_res)
            )
            profile.update({
                'crs': src.crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            os.makedirs(dst_path.parent, exist_ok=True)
            with rasterio.open(dst_path, 'w', **profile) as dst:
                for b in range(1, src.count + 1):
                    band = src.read(
                        b,
                        out_shape=(height, width),
                        resampling=resampling_method
                    )
                    dst.write(band, b)


def resample_image(src_path: Path, dst_path: Path, target_res: float, orig_res: float = 0.05, resample_method=Image.BILINEAR):
    """
    Resample a non-georeferenced image (e.g., JPG) to the target resolution using PIL.
    """
    img = Image.open(src_path)
    scale = orig_res / target_res
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)
    img_resized = img.resize((new_width, new_height), resample_method)
    os.makedirs(dst_path.parent, exist_ok=True)
    img_resized.save(dst_path)


def main():
    src_dir = Path('/home/p24030854116/datebase/ISPRS_dataset/Potsdam')
    dst_dir = Path('/home/p24030854116/datebase/ISPRS_dataset2/Potsdam')
    target_res = 0.09  # 9 cm
    orig_res = 0.05    # 5 cm

    # Walk through all relevant files
    for src_path in src_dir.rglob('*'):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel

        # Skip if already processed
        if dst_path.exists():
            print(f"Skipping existing file: {dst_path}")
            continue

        # Process GeoTIFFs
        if src_path.suffix.lower() in ['.tif', '.tiff']:
            # Choose interpolation based on folder
            if 'Labels' in src_path.parts:
                method = Resampling.nearest
            else:
                method = Resampling.bilinear
            print(f"[GeoTIFF] {src_path.name} -> {dst_path} ({method.name})")
            resample_raster(src_path, dst_path, target_res, method, orig_res)

        # Process images
        elif src_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"[Image] {src_path.name} -> {dst_path} (PIL resize)")
            resample_image(src_path, dst_path, target_res, orig_res, Image.BILINEAR)

if __name__ == '__main__':
    main()
