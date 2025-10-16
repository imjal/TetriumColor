from typing import List, Tuple

from PIL import Image, ImageDraw
import math


def CreateCircleGridImages(colors: List[Tuple | List[int]],
                           grid_size: int = None,
                           circle_radius: int = 50,
                           padding: int = 10,
                           bg_color=(0, 0, 0), out_prefix="output"):
    """
    Create nxn grid images of circles colored from 'colors' list (rgb tuples 0-255), 
    If len(colors) > 4, produce image(s) with up to 4 colors each, as _RGB.png and _OCV.png

    Args:
        colors (list): a list of color tuples, e.g., [(255,0,0), ...] (should be valid Pillow colors)
        grid_size (int, optional): if None, auto-compute for closest square
        circle_radius (int): radius of circles drawn
        padding (int): pixel padding between circles
        bg_color (tuple): background color
        out_prefix (str): prefix for output files

    Returns:
        If len(colors) <= 4: returns the created grid (Pillow Image)
        Otherwise: saves files ("{out_prefix}_{i}_RGB.png", "{out_prefix}_{i}_OCV.png") and returns the list of image files
    """

    # default grid size
    n = grid_size if grid_size is not None else math.ceil(math.sqrt(len(colors)))
    img_side = 2 * circle_radius + padding
    img_width = n * img_side + padding
    img_height = n * img_side + padding

    def make_grid(colors_block):
        img = Image.new("RGB", (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(img)
        for idx, color in enumerate(colors_block):
            row = idx // n
            col = idx % n
            cx = padding + col * img_side + circle_radius
            cy = padding + row * img_side + circle_radius
            bbox = [cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius]
            draw.ellipse(bbox, fill=tuple(color))
        return img

    results = []
    if colors.shape[1] > 4:
        # Make images each with up to 4 colors at a time (for plate format)
        path_ends = ["_RGB.png", "_OCV.png"]
        blocks = [colors[:, i:i+3] for i in range(2)]
        img_blocks = []
        for i, block in enumerate(blocks):
            # Build block image (grid will be at most 2x2)
            img_blocks.append(make_grid(block))
            # Save as _RGB.png and _OCV.png
        return tuple(img_blocks)
    else:
        img = make_grid(colors)
        return img


def CreatePaddedGrid(images: List[str] | List[Image.Image], canvas_size=(1280, 720), padding=10, bg_color=(0, 0, 0), channels=3, square_grid=True) -> Image.Image:
    """
    Create a padded grid of images centered on a canvas of specified size.

    Args:
        images (list of str or list of Image.Image): List of image file paths or Pillow Image objects.
        canvas_size (tuple): Tuple (width, height) specifying the canvas dimensions.
        padding (int, optional): Padding between images in pixels. Defaults to 10.
        bg_color (tuple, optional): Background color for the canvas (R, G, B). Defaults to black.

    Returns:
        Image: The grid as a Pillow Image object centered on the canvas.
    """
    # Load all images
    if isinstance(images[0], str):
        images = [Image.open(file) for file in images]

    # Ensure all images are the same size baseline (we will resize again to optimal square size)
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Dynamically compute grid size to best fit N square images inside the canvas.
    num_images = len(images)
    canvas_width, canvas_height = canvas_size

    def best_cell_size_for(cols: int, rows: int) -> int:
        # available space minus paddings
        if cols <= 0 or rows <= 0:
            return 0
        total_pad_x = padding * (cols - 1)
        total_pad_y = padding * (rows - 1)
        if canvas_width <= total_pad_x or canvas_height <= total_pad_y:
            return 0
        cell_w = (canvas_width - total_pad_x) / cols
        cell_h = (canvas_height - total_pad_y) / rows
        return int(max(0, math.floor(min(cell_w, cell_h))))

    # Choose grid shape
    if square_grid:
        cols = rows = math.ceil(math.sqrt(num_images))
    else:
        # Search all reasonable column counts; pick the layout maximizing cell size
        best = (0, 0, 0)  # (cell_size, cols, rows)
        for c in range(1, num_images + 1):
            r = math.ceil(num_images / c)
            s = best_cell_size_for(c, r)
            # Prefer larger cell size; tie-breaker: smaller grid area
            if s > best[0] or (s == best[0] and c * r < best[1] * best[2] if best[1] and best[2] else False):
                best = (s, c, r)
        cell_size, cols, rows = best

    if cell_size <= 0:
        raise ValueError("Canvas too small to fit images with given padding")

    # Calculate grid dimensions using the optimal square cell size
    grid_width = cols * cell_size + (cols - 1) * padding
    grid_height = rows * cell_size + (rows - 1) * padding

    if channels == 4:
        mode = "RGBA"
    elif channels == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    # Create a blank canvas
    canvas = Image.new(mode, (canvas_width, canvas_height), bg_color)

    # Create the grid
    grid_image = Image.new(mode, (grid_width, grid_height), bg_color)
    # Resize images to square cells
    resized_images = [img.resize((cell_size, cell_size)) for img in images]
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = col * (cell_size + padding)
        y = row * (cell_size + padding)
        grid_image.paste(img, (x, y))

    # Center the grid on the canvas
    x_offset = (canvas_width - grid_width) // 2
    y_offset = (canvas_height - grid_height) // 2
    canvas.paste(grid_image, (x_offset, y_offset))

    return canvas


def ExportPlates(images: List[Tuple[Image.Image, Image.Image]], filename: str):
    """
    Export a list of images as a padded grid to a file.

    Args:
        images (list of Image.Image): List of Pillow Image objects.
        filename (str): The output file path.
        canvas_size (tuple): Tuple (width, height) specifying the canvas dimensions.
        padding (int, optional): Padding between images in pixels. Defaults to 10.
        bg_color (tuple, optional): Background color for the canvas (R, G, B). Defaults to black.
    """
    img_rgo = CreatePaddedGrid([i[0] for i in images], padding=0)
    img_rgo.save(f"{filename}_RGB.png")
    img_bgo = CreatePaddedGrid([i[1] for i in images], padding=0)
    img_bgo.save(f"{filename}_OCV.png")
