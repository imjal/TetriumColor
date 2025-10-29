import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple
from TetriumColor.ColorSpace import ColorSpace, ColorSpaceType


class SolidColorGenerator:
    """Generates solid color circle stimuli for psychophysics experiments."""
    
    def __init__(self, color_space: ColorSpace):
        """
        Initialize the generator with a color space for conversion.
        
        Args:
            color_space: ColorSpace object for RGBO -> DISP_6P conversion
        """
        self.color_space = color_space
    
    def generate_circle(
        self,
        filename_base: str,
        rgbo_values: Tuple[float, float, float, float],
        image_size: int = 512,
        circle_radius_ratio: float = 0.8,
        has_noisy_boundary: bool = False,
        output_space: ColorSpaceType = ColorSpaceType.DISP_6P
    ) -> Tuple[str, str]:
        """
        Generate a solid color circle and save as RGB and OCV images.
        
        Args:
            filename_base: Base filename (without extension)
            rgbo_values: RGBO color values (0-255 range)
            image_size: Size of the output image in pixels
            circle_radius_ratio: Circle radius as ratio of image size (0-1)
            has_noisy_boundary: Whether to add noise to the circle boundary
            output_space: Output color space (default DISP_6P)
            
        Returns:
            Tuple of (rgb_path, ocv_path)
        """
        # Normalize RGBO to 0-1 range
        rgbo_normalized = np.array(rgbo_values) / 255.0
        
        # Convert RGBO to DISP color space (single point conversion)
        rgbo_point = rgbo_normalized.reshape(1, 4)
        disp_color = self.color_space.convert(
            rgbo_point,
            from_space=ColorSpaceType.DISP,
            to_space=output_space
        )[0]  # Get first (and only) point
        
        # Split into RGB and OCV channels
        if output_space == ColorSpaceType.DISP_6P:
            rgb_color = disp_color[:3]  # Channels 0, 1, 2
            ocv_color = disp_color[3:]  # Channels 3, 4, 5
        else:
            # For SRGB output, both are the same
            rgb_color = disp_color
            ocv_color = disp_color
        
        # Clip and convert to 8-bit
        rgb_color_8bit = np.clip(rgb_color * 255, 0, 255).astype(np.uint8)
        ocv_color_8bit = np.clip(ocv_color * 255, 0, 255).astype(np.uint8)
        
        # Generate the circle images
        rgb_img = self._create_circle_image(
            image_size,
            circle_radius_ratio,
            tuple(rgb_color_8bit),
            has_noisy_boundary
        )
        
        ocv_img = self._create_circle_image(
            image_size,
            circle_radius_ratio,
            tuple(ocv_color_8bit),
            has_noisy_boundary
        )
        
        # Save images
        rgb_path = filename_base + "_RGB.png"
        ocv_path = filename_base + "_OCV.png"
        
        rgb_img.save(rgb_path)
        ocv_img.save(ocv_path)
        
        return rgb_path, ocv_path
    
    def _create_circle_image(
        self,
        size: int,
        radius_ratio: float,
        color: Tuple[int, int, int],
        has_noisy_boundary: bool
    ) -> Image.Image:
        """
        Create a single circle image.
        
        Args:
            size: Image size in pixels
            radius_ratio: Circle radius as ratio of image size
            color: RGB color tuple (0-255)
            has_noisy_boundary: Whether to add noise to boundary
            
        Returns:
            PIL Image object
        """
        # Create black background
        img = Image.new("RGB", (size, size), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Calculate circle bounds
        center = size // 2
        radius = int(size * radius_ratio / 2)
        
        bbox = [
            center - radius,
            center - radius,
            center + radius,
            center + radius
        ]
        
        # Draw the solid circle
        draw.ellipse(bbox, fill=color)
        
        # Optional: Add noisy boundary
        if has_noisy_boundary:
            # Simple noise implementation: add random pixels near the edge
            pixels = img.load()
            for y in range(size):
                for x in range(size):
                    dx = x - center
                    dy = y - center
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Add noise in a band near the radius
                    if abs(dist - radius) < 5:
                        noise = np.random.randint(-20, 20, 3)
                        noisy_color = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
                        pixels[x, y] = tuple(noisy_color)
        
        return img

