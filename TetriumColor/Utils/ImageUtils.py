from typing import List, Tuple

from PIL import Image
import math


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
