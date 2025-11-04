from PIL import Image
import numpy as np


def create_gradient_bar(width, height, start_color, end_color):
    """Create a gradient bar from start_color to end_color."""
    gradient = Image.new("RGB", (width, height))
    for x in range(width):
        r = int(start_color[0] + (end_color[0] - start_color[0]) * (x / (width - 1)))
        g = int(start_color[1] + (end_color[1] - start_color[1]) * (x / (width - 1)))
        b = int(start_color[2] + (end_color[2] - start_color[2]) * (x / (width - 1)))
        for y in range(height):
            gradient.putpixel((x, y), (r, g, b))
    return gradient


def create_image(filename, bars):
    """Create an image with multiple gradient bars."""
    bar_width = 256
    bar_height = 50
    image_height = bar_height * len(bars)
    image = Image.new("RGB", (bar_width, image_height))

    for i, color in enumerate(bars):
        bar = create_gradient_bar(bar_width, bar_height, [0, 0, 0],  color)
        image.paste(bar, (0, i * bar_height))

    image.save(filename)


# First image
bars = np.array([
    ([255, 0, 0], [0, 0, 0]),  # First bar
    ([0, 255, 0], [0, 255, 0]),  # Second bar
    ([0, 0, 255], [0, 0, 255]),  # Third bar
    ([0, 0, 0], [255, 0, 0])   # Fourth bar
])
create_image("./assets/grid_RGB.png", bars[:, 0])
create_image("./assets/grid_OCV.png", bars[:, 1])


blank = np.array([[114, 169, 191], [96, 169, 191]])


def create_blank_image(rgb_value, width, height):
    """Create a blank image with a specific RGB value."""
    image = Image.new("RGB", (width, height), tuple(rgb_value))
    return image


# Create blank images
blank_image_1 = create_blank_image([114, 169, 191], 256, 256)
blank_image_2 = create_blank_image([96, 169, 191], 256, 256)

# Save blank images
blank_image_1.save("./assets/blank_RGB.png")
blank_image_2.save("./assets/blank_OCV.png")
