#!/usr/bin/env python3
"""
Generate RGB.png and OCV.png calibration images.

The images have 4 rows, each divided into 14 rectangles with channel values:
1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128

RGB.png is an RGO image (Red, Green, Orange channels)
OCV.png is a BGO image (Blue, Green, Orange channels)
"""

from PIL import Image
import numpy as np

# Channel values for each rectangle
channel_values = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128]
num_rectangles = len(channel_values)

# Image dimensions
rect_width = 100  # Width of each rectangle
rect_height = 100  # Height of each row
gap_width = 4  # Width of black strip between rectangles
num_rows = 4

# Calculate image width accounting for gaps every 2 blocks
# Gaps occur after every pair of blocks (after indices 1, 3, 5, 7, 9, 11, 13)
num_gaps = num_rectangles // 2
image_width = num_rectangles * rect_width + num_gaps * gap_width
image_height = num_rows * rect_height


def create_rgo_image():
    """Create RGB.png with RGO channels (Red, Green, Orange).

    The 4 rows represent RGBO channels:
    - Row 0: Red channel
    - Row 1: Green channel  
    - Row 2: Blue channel (not used in RGO, left black)
    - Row 3: Orange channel
    """
    # Create a 3-channel image (RGB for display)
    # We'll encode RGO as RGB channels
    img_array = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for row_idx in range(num_rows):
        y_start = row_idx * rect_height
        y_end = y_start + rect_height

        for col_idx, value in enumerate(channel_values):
            # Calculate x position accounting for gaps every 2 blocks
            # Gap occurs after every pair (after indices 1, 3, 5, etc.)
            gaps_before = col_idx // 2  # Number of gaps before this block
            x_start = col_idx * rect_width + gaps_before * gap_width
            x_end = x_start + rect_width

            if row_idx == 0:  # Red channel (R)
                img_array[y_start:y_end, x_start:x_end, 0] = value  # R
            elif row_idx == 1:  # Green channel (G)
                img_array[y_start:y_end, x_start:x_end, 1] = value  # G
            elif row_idx == 2:  # Blue channel (B) - not used in RGO, leave black
                pass
            else:  # Row 3: Orange channel (O)
                # Orange in RGB space: use red channel with orange tint
                img_array[y_start:y_end, x_start:x_end, 2] = value  # R (for orange)

    return Image.fromarray(img_array, 'RGB')


def create_bgo_image():
    """Create OCV.png with BGO channels (Blue, Green, Orange).

    The 4 rows represent RGBO channels:
    - Row 0: Red channel (not used in BGO, left black)
    - Row 1: Green channel
    - Row 2: Blue channel
    - Row 3: Orange channel
    """
    # Create a 3-channel image (RGB for display)
    # We'll encode BGO as RGB channels
    img_array = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for row_idx in range(num_rows):
        y_start = row_idx * rect_height
        y_end = y_start + rect_height

        for col_idx, value in enumerate(channel_values):
            # Calculate x position accounting for gaps every 2 blocks
            # Gap occurs after every pair (after indices 1, 3, 5, etc.)
            gaps_before = col_idx // 2  # Number of gaps before this block
            x_start = col_idx * rect_width + gaps_before * gap_width
            x_end = x_start + rect_width

            if row_idx == 0:  # Red channel (R) - not used in BGO, leave black
                pass
            elif row_idx == 1:  # Green channel (G)
                img_array[y_start:y_end, x_start:x_end, 1] = value  # G
            elif row_idx == 2:  # Blue channel (B)
                img_array[y_start:y_end, x_start:x_end, 0] = value  # B
            elif row_idx == 3:  # Row 3: Orange channel (O)
                # Orange in RGB space: use red channel with orange tint
                img_array[y_start:y_end, x_start:x_end, 2] = value  # R (for orange)

    return Image.fromarray(img_array, 'RGB')


def main():
    print("Generating RGB.png (RGO image)...")
    rgb_img = create_rgo_image()
    rgb_img.save("assets/textures/RGB.png")
    print(f"Saved RGB.png: {rgb_img.size[0]}x{rgb_img.size[1]}")

    print("Generating OCV.png (BGO image)...")
    ocv_img = create_bgo_image()
    ocv_img.save("assets/textures/OCV.png")
    print(f"Saved OCV.png: {ocv_img.size[0]}x{ocv_img.size[1]}")

    print("\nImage layout:")
    print(f"  - {num_rows} rows x {num_rectangles} rectangles")
    print(f"  - Rectangle size: {rect_width}x{rect_height} pixels")
    print(f"  - Gap every 2 blocks: {gap_width} pixels (black strip)")
    print(f"  - Channel values: {channel_values}")
    print(f"  - RGB.png (RGO): Row 0=Red, Row 1=Green, Row 2=Black, Row 3=Orange")
    print(f"  - OCV.png (BGO): Row 0=Black, Row 1=Green, Row 2=Blue, Row 3=Orange")


if __name__ == "__main__":
    main()
