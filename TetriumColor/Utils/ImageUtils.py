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


def CreatePaddedGrid(images: List[str] | List[Image.Image], canvas_size=(1280, 720), padding=10, bg_color=(0, 0, 0), channels=3) -> Image.Image:
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

    # Ensure all images are the same size
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    resized_images = [img.resize((max_width, max_height)) for img in images]

    # Determine grid size (square grid)
    num_images = len(images)
    cols = rows = math.ceil(math.sqrt(num_images))

    # Calculate grid dimensions
    grid_width = cols * max_width + (cols - 1) * padding
    grid_height = rows * max_height + (rows - 1) * padding

    if channels == 4:
        mode = "RGBA"
    elif channels == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    # Create a blank canvas
    canvas_width, canvas_height = canvas_size
    canvas = Image.new(mode, (canvas_width, canvas_height), bg_color)

    # Create the grid
    grid_image = Image.new(mode, (grid_width, grid_height), bg_color)
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = col * (max_width + padding)
        y = row * (max_height + padding)
        grid_image.paste(img, (x, y))

    grid_image = grid_image.resize((canvas_size[1], canvas_size[1]))
    # Center the grid on the canvas
    x_offset = (canvas_width - canvas_size[1]) // 2
    y_offset = (canvas_height - canvas_size[1]) // 2
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
