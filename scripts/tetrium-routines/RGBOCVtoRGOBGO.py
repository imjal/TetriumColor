from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


def read_png(path: Path) -> np.ndarray:
    """Read a PNG as an ndarray with shape (H, W, 3).

    Preserves the underlying dtype (typically uint8 or uint16). Converts
    paletted/greyscale PNGs to RGB if encountered.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image with 3 channels at {path}")
    return arr


def write_png(path: Path, arr: np.ndarray) -> None:
    """Write an ndarray (H, W, 3) to PNG, preserving dtype when possible."""
    img = Image.fromarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def find_pairs(root: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield (rgb_path, ocv_path) for every matching prefix in the tree.

    A pair is recognized when both `*_RGB.png` and `*_OCV.png` exist with the
    same leading stem before the suffix.
    """
    rgb_files = list(root.rglob("*_RGB.png"))
    ocv_index = {p.with_name(p.name.replace("_OCV.png", "")): p for p in root.rglob("*_OCV.png")}
    for rgb in rgb_files:
        key = rgb.with_name(rgb.name.replace("_RGB.png", ""))
        ocv = ocv_index.get(key)
        if ocv is not None:
            yield rgb, ocv


def remap_channels(rgb_img: np.ndarray, ocv_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create RGO and BGO images.

    - RGO: [R_from_RGB, G_from_RGB, O_from_OCV]
    - BGO: [B_from_RGB, G_from_RGB, O_from_OCV]
    """
    if rgb_img.shape != ocv_img.shape:
        raise ValueError(
            f"Image size/channel mismatch: RGB {rgb_img.shape} vs OCV {ocv_img.shape}"
        )

    # Channels from RGB
    r = rgb_img[:, :, 0]
    g = rgb_img[:, :, 1]
    b = rgb_img[:, :, 2]

    # Interpret OCV stored in RGB order as channels [O, C, V]
    o = ocv_img[:, :, 0]

    rgo = np.stack([r, g, o], axis=2)
    bgo = np.stack([b, g, o], axis=2)
    return rgo, bgo


def process_pair(rgb_path: Path, ocv_path: Path, out_dir: Path | None, overwrite: bool) -> Tuple[Path, Path]:
    rgb_img = read_png(rgb_path)
    ocv_img = read_png(ocv_path)
    rgo, bgo = remap_channels(rgb_img, ocv_img)

    base_stem = rgb_path.name.replace("_RGB.png", "")
    if out_dir is None:
        out_dir = rgb_path.parent

    rgo_path = out_dir / f"{base_stem}_RGB.png"
    bgo_path = out_dir / f"{base_stem}_OCV.png"

    if not overwrite and (rgo_path.exists() or bgo_path.exists()):
        # Avoid accidental overwrite unless explicitly requested
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {rgo_path} or {bgo_path}")

    write_png(rgo_path, rgo)
    write_png(bgo_path, bgo)
    return rgo_path, bgo_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remap paired *_RGB.png and *_OCV.png into *_RGO.png and *_BGO.png, "
            "discarding C and V channels from OCV and keeping O."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Root directory to scan recursively (default: current directory)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory; mirrors input structure beneath this directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_RGO.png and *_BGO.png if present.",
    )

    args = parser.parse_args()

    pairs = list(find_pairs(args.root))
    if not pairs:
        print("No *_RGB.png and *_OCV.png pairs found.")
        return

    total = 0
    for rgb_path, ocv_path in pairs:
        if args.out_dir is not None:
            # Recreate relative structure under the output directory
            rel_parent = rgb_path.parent.relative_to(args.root)
            target_dir = args.out_dir / rel_parent
        else:
            target_dir = None

        rgo_path, bgo_path = process_pair(rgb_path, ocv_path, target_dir, args.overwrite)
        print(f"Wrote {rgo_path}")
        print(f"Wrote {bgo_path}")
        total += 2

    print(f"Done. Wrote {total} files across {len(pairs)} pairs.")


if __name__ == "__main__":
    main()
