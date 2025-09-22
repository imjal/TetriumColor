from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy.typing as npt

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector


try:
    import spectral as sp
except ImportError as e:
    raise ImportError(
        "The 'spectral' library is required for hyperspectral image operations. Install via 'pip install spectral' ") from e

from TetriumColor.Observer.Spectra import Spectra


def _parse_wavelengths_from_metadata(md: dict, bands_count: int) -> Optional[np.ndarray]:
    """Attempt to parse wavelengths from various common metadata keys."""
    candidates = [
        "wavelength",
        "wavelengths",
        "band centers",
        "bands",
        "centers",
    ]
    for key in candidates:
        if key in md and md[key] is not None:
            vals = md[key]
            # ENVI often stores as list[str] or list[float]
            if isinstance(vals, str):
                # Could be a comma/space separated list
                parts = [p.strip() for p in vals.replace("{", "").replace("}", "").split(',') if p.strip()]
                try:
                    arr = np.array([float(p) for p in parts], dtype=float)
                except ValueError:
                    continue
            elif isinstance(vals, (list, tuple)):
                try:
                    arr = np.array([float(v) for v in vals], dtype=float)
                except Exception:
                    continue
            else:
                continue
            if arr.ndim == 1 and arr.size == bands_count:
                return arr
    # Some formats expose via image.bands.centers but not metadata dict
    return None


def _get_wavelengths(image) -> np.ndarray:
    """Extract wavelength array from a spectral image, with reasonable fallbacks."""
    bands_count = image.shape[-1]

    # Try image metadata
    md = getattr(image, 'metadata', None)
    if isinstance(md, dict):
        arr = _parse_wavelengths_from_metadata(md, bands_count)
        if arr is not None:
            return arr

    # Try image.bands.centers
    bands = getattr(image, 'bands', None)
    centers = getattr(bands, 'centers', None) if bands is not None else None
    if centers is not None:
        arr = np.array(centers, dtype=float)
        if arr.size == bands_count:
            return arr

    # Fallback to 400-700 nm linear if nothing found
    return np.linspace(400.0, 700.0, bands_count, dtype=float)


def extract_grid_from_image(filename_envi: str, filename_rgb: str, halftones: List[tuple], debug: bool = True) -> Dict[str, Spectra]:
    """Extract average spectra for each colored square using the RGB image to find cells.

    Steps:
    - Use the RGB image to detect colored squares (white gutters ignored).
    - Map each detected RGB bounding box into hyperspectral coordinates by scaling.
    - Average spectra inside each mapped box and assign to the provided halftones (row-major).

    Args:
        filename_envi: Path to hyperspectral image file
        filename_rgb: Path to RGB reference image
        halftones: List of halftone tuples to assign to detected cells
        debug: If True, display debugging visualization showing detected rectangles
    """
    # Load RGB
    rgb = cv2.imread(filename_rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Could not open RGB image: {filename_rgb}")
    rgb_h, rgb_w = rgb.shape[:2]

    # Convert BGR to RGB for matplotlib display
    rgb_display = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Convert to HSV and create mask for non-white (colored) regions
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Threshold: white tends to have low saturation; keep pixels with enough saturation and brightness
    s_thresh = 30
    v_min = 40
    color_mask = cv2.inRange(hsv, (0, s_thresh, v_min), (179, 255, 255))

    # Morphology to clean and fill
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find connected components (contours) for squares
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []  # (x0, y0, x1, y1)
    min_area = 0.0005 * (rgb_h * rgb_w)  # filter tiny noise
    for cnt in contours:
        x, y, w, h_ = cv2.boundingRect(cnt)
        area = w * h_
        if area < min_area:
            continue
        boxes.append((x, y, x + w, y + h_))

    # If count mismatches halftones, derive a regular grid inside the mask's bounding box
    expected = len(halftones)
    if len(boxes) != expected and expected > 0:
        ys, xs = np.where(color_mask > 0)
        if ys.size == 0:
            raise ValueError("No colored regions detected in RGB image.")
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        # Factor expected into rows x cols close to square

        def _factor_grid(n: int) -> Tuple[int, int]:
            best = (1, n)
            for r in range(1, int(np.sqrt(n)) + 1):
                if n % r == 0:
                    c = n // r
                    if abs(c - r) < abs(best[1] - best[0]):
                        best = (r, c)
            return best
        rows, cols = _factor_grid(expected)
        xs_edges = np.linspace(x0, x1 + 1, cols + 1, dtype=int)
        ys_edges = np.linspace(y0, y1 + 1, rows + 1, dtype=int)
        boxes = []
        for r in range(rows):
            for c in range(cols):
                bx0, bx1 = xs_edges[c], xs_edges[c + 1]
                by0, by1 = ys_edges[r], ys_edges[r + 1]
                boxes.append((bx0, by0, bx1, by1))

    if len(boxes) != expected:
        raise ValueError(f"Detected {len(boxes)} cells from RGB, but {expected} halftones were provided.")

    # Sort boxes row-major (by y, then x)
    def _sort_key(b):
        x0, y0, x1, y1 = b
        return (y0, x0)
    boxes.sort(key=_sort_key)

    # DEBUG VISUALIZATION
    if debug:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original RGB image
        axes[0, 0].imshow(rgb_display)
        axes[0, 0].set_title("Original RGB Image")
        axes[0, 0].axis('off')

        # Color mask
        axes[0, 1].imshow(color_mask, cmap='gray')
        axes[0, 1].set_title("Color Detection Mask")
        axes[0, 1].axis('off')

        # RGB with detected boxes (original detection)
        axes[1, 0].imshow(rgb_display)
        axes[1, 0].set_title("Detected Rectangles (Original)")
        axes[1, 0].axis('off')

        # RGB with final sorted boxes
        axes[1, 1].imshow(rgb_display)
        axes[1, 1].set_title("Final Sorted Rectangles (Row-Major)")
        axes[1, 1].axis('off')

        # Draw rectangles on both bottom plots
        for idx, box in enumerate(boxes):
            x0, y0, x1, y1 = box

            # Draw on original detection plot (bottom left)
            from matplotlib.patches import Rectangle
            rect1 = Rectangle((x0, y0), x1-x0, y1-y0,
                              linewidth=2, edgecolor='red', facecolor='none')
            axes[1, 0].add_patch(rect1)

            # Draw on final sorted plot (bottom right) with numbers
            rect2 = Rectangle((x0, y0), x1-x0, y1-y0,
                              linewidth=2, edgecolor='lime', facecolor='none')
            axes[1, 1].add_patch(rect2)

            # Add text with index number
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            axes[1, 1].text(cx, cy, str(idx), ha='center', va='center',
                            color='lime', fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

            # Also shrink rectangles to show the actual sampling area
            dx = max(1, (x1 - x0) // 20)
            dy = max(1, (y1 - y0) // 20)
            x0i, x1i = x0 + dx, x1 - dx
            y0i, y1i = y0 + dy, y1 - dy
            if x1i > x0i and y1i > y0i:
                rect3 = Rectangle((x0i, y0i), x1i-x0i, y1i-y0i,
                                  linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--')
                axes[1, 1].add_patch(rect3)

        plt.tight_layout()
        plt.show()

        print(f"Detected {len(boxes)} rectangles for {len(halftones)} expected halftones")
        print("Rectangle coordinates (x0, y0, x1, y1):")
        for i, box in enumerate(boxes):
            print(f"  {i}: {box} -> halftone {halftones[i]}")

    # Load hyperspectral image and wavelengths
    img = sp.open_image(filename_envi)
    cube = img.load().astype(np.float32)  # (rows, cols, bands)
    cube = np.transpose(cube, (1, 0, 2))[:, ::-1, :]
    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube (rows, cols, bands).")
    hs_h, hs_w, _ = cube.shape

    try:
        wavelengths = _get_wavelengths(img)  # use existing helper
    except NameError:
        # Fallback if helper not available in this context
        bands_count = cube.shape[-1]
        wavelengths = np.linspace(400.0, 700.0, bands_count, dtype=float)

    # Scale factors from RGB -> HS coordinate space
    sy = hs_h / float(rgb_h)
    sx = hs_w / float(rgb_w)

    if debug:
        print(f"Hyperspectral image size: {hs_h}x{hs_w} (vs RGB: {rgb_h}x{rgb_w})")
        print(f"Scale factors: sx={sx:.3f}, sy={sy:.3f}")

    spectra_map: Dict[str, Spectra] = {}

    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        # shrink a bit to avoid gutters bleeding
        dx = max(1, (x1 - x0) // 20)
        dy = max(1, (y1 - y0) // 20)
        x0i, x1i = x0 + dx, x1 - dx
        y0i, y1i = y0 + dy, y1 - dy
        if x1i <= x0i or y1i <= y0i:
            x0i, y0i, x1i, y1i = x0, y0, x1, y1

        # Map to HS coordinates
        hx0 = int(np.clip(round(x0i * sx), 0, hs_w - 1))
        hx1 = int(np.clip(round(x1i * sx), 1, hs_w))
        hy0 = int(np.clip(round(y0i * sy), 0, hs_h - 1))
        hy1 = int(np.clip(round(y1i * sy), 1, hs_h))

        if debug and idx < 5:  # Print details for first few boxes
            print(f"Box {idx}: RGB({x0i},{y0i},{x1i},{y1i}) -> HS({hx0},{hy0},{hx1},{hy1})")

        tile = cube[hy0:hy1, hx0:hx1, :]
        if tile.size == 0:
            raise ValueError(f"Empty HS tile for cell {idx} mapped from RGB box {box}.")

        mean_spec = tile.reshape(-1, tile.shape[-1]).mean(axis=0)
        data = np.clip(mean_spec.astype(float), 0.0, 1.0)
        spectra = Spectra(wavelengths=wavelengths, data=data, normalized=True)

        key = str(halftones[idx])
        spectra_map[key] = spectra

    return spectra_map


def _closest_band_indices(wavelengths: np.ndarray, targets: List[float]) -> List[int]:
    idxs: List[int] = []
    for t in targets:
        idxs.append(int(np.argmin(np.abs(wavelengths - t))))
    return idxs


def inspect_hyperspectral_image(filename: str, output_wavelengths: Optional[npt.NDArray] = np.arange(400, 700, 1)) -> List[Spectra]:
    """Interactive inspector using matplotlib.

    - Displays an RGB composite of the hyperspectral cube (no spectral.imshow used).
    - Left-click anywhere to sample that pixel's spectrum; plots it live and stores it.
    - Click-and-drag to draw a rectangle; on release, averages spectra in the region; plots and stores it.
    - Close the window to return the list of collected Spectra objects (in order of selection).
    """
    img = sp.open_image(filename)
    cube = img.load()
    cube = np.transpose(cube, (0, 1, 2))[:, :, :]
    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube (rows, cols, bands).")

    wavelengths = _get_wavelengths(img)

    # Build RGB composite from chosen bands (approx 650/550/450 nm)
    if wavelengths is not None and wavelengths.size == cube.shape[-1]:
        r_idx, g_idx, b_idx = _closest_band_indices(wavelengths, [650.0, 550.0, 450.0])
    else:
        bands = cube.shape[-1]
        r_idx, g_idx, b_idx = bands - 1, bands // 2, 0

    rgb = np.stack([cube[..., r_idx], cube[..., g_idx], cube[..., b_idx]], axis=-1).astype(np.float32)
    # Normalize per-channel to 0-1 for display
    for c in range(3):
        ch = rgb[..., c]
        ch_min, ch_max = float(ch.min()), float(ch.max())
        if ch_max > ch_min:
            rgb[..., c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            rgb[..., c] = 0.0

    # Prepare interactive figure: image on left, spectra on right
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 6))
    im = ax_img.imshow(rgb)
    ax_img.set_title("Hyperspectral RGB composite (click or drag to select)")
    ax_img.set_axis_off()

    ax_spec.set_title("Selected spectra")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Relative intensity")

    collected: List[Spectra] = []

    # Helper to plot a spectra
    def plot_spectra(s: Spectra, label: Optional[str] = None):
        color = None
        try:
            color = s.to_rgb()
        except Exception:
            pass
        ax_spec.plot(s.wavelengths, s.data / (np.max(s.data) if np.max(s.data) > 0 else 1), label=label, color=color)
        ax_spec.legend(loc="best", fontsize=8)
        ax_spec.figure.canvas.draw_idle()

    # Click handler: sample single pixel
    def on_click(event):
        if event.inaxes != ax_img:
            return
        if event.button != 1:  # left-click
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        h, w, _ = cube.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        spectrum = cube[y, x, :].astype(np.float32)
        data = np.clip(spectrum, 0.0, 1.0)
        s = Spectra(wavelengths=wavelengths, data=data, normalized=True)
        new_spectra = s.interpolate(output_wavelengths)
        collected.append(new_spectra)
        plot_spectra(new_spectra, label=f"({x},{y})")

    # Rectangle selector: average within ROI
    start_end = {"x0": None, "y0": None}

    def on_select(eclick, erelease):
        x0, y0 = int(np.floor(eclick.xdata)), int(np.floor(eclick.ydata))
        x1, y1 = int(np.ceil(erelease.xdata)), int(np.ceil(erelease.ydata))
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        h, w, _ = cube.shape
        x0 = max(0, min(w - 1, x0))
        x1 = max(1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(1, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            return
        tile = cube[y0:y1, x0:x1, :]
        mean_spec = tile.reshape(-1, tile.shape[-1]).mean(axis=0)
        data = np.clip(mean_spec.astype(np.float32), 0.0, 1.0)
        s = Spectra(wavelengths=wavelengths, data=data, normalized=True)
        new_spectra = s.interpolate(output_wavelengths)
        collected.append(new_spectra)
        plot_spectra(new_spectra, label=f"[{x0}:{x1},{y0}:{y1}]")

    rs = RectangleSelector(
        ax_img,
        on_select,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords='pixels',
        interactive=True,
    )

    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show(block=True)

    # Cleanup handlers (not strictly necessary, but good practice)
    try:
        fig.canvas.mpl_disconnect(cid)
        rs.disconnect_events()
    except Exception:
        pass

    return collected
