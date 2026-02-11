import numpy as np
from PIL import Image
import tifffile


# Define the 4x4 transformation matrices
mat_bgyr_to_rgb = np.array([
 [ 0.982,-0.021,-0.01 ,0],
 [-0.083, 0.092,-0.011,0],
 [ 0.099, 0.825, 0.083,0],
 [ 0.002, 0.02,  0.841,0],
])

mat_bgyr_to_ocv = np.array(
[[ 0.068, 0.,    0.  ,0],
 [ 0.955, 0.,    0.  ,0],
 [-0.304, 0.,    0.  ,0],
 [-0.035, 0.,    0.  ,0]]
)

def load_image(image_path):
    # Load the TIFF image using tifffile
    img = tifffile.imread(image_path)

    # interpret byte as floats instead
    img = img.view(np.float32)

    return img

def apply_matrix_to_pixels(img_data, matrix):
    # The image data is assumed to be in the format of 4 channels per pixel (BGYR)
    height, width, _ = img_data.shape
    #print(img_data)
    transformed_data = np.zeros_like(img_data, dtype=np.float32)

    # Apply the matrix to each pixel (each pixel has 4 channels)
    for y in range(height):
        for x in range(width):
            pixel = img_data[y, x]  # Get the 4 channels of the pixel
            transformed_pixel = np.dot(matrix.T, pixel)  # Apply the matrix transformation
            transformed_data[y, x] = transformed_pixel
    #print(transformed_data)
    
    return transformed_data

def normalize_and_save_image(img_data, output_path):
    # Set the 4th channel (alpha) to 1
    img_data[:, :, 3] = 1.0

    #print(img_data)
    # Scale the pixel values by 255
    img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)

    #img_data = np.clip(img_data, 0, 1).astype(np.float32)
    
    # Convert the numpy array back to an image and save it as PNG
    img_out = Image.fromarray(img_data)
    img_out.save(output_path)

def main(input_path, output_rgb_path, output_ocv_path):
    # Load the input image
    img_data = load_image(input_path)
    
    # Apply the transformation matrices
    rgb_data = apply_matrix_to_pixels(img_data, mat_bgyr_to_rgb)
    ocv_data = apply_matrix_to_pixels(img_data, mat_bgyr_to_ocv)
    
    # Normalize and save both transformed images
    normalize_and_save_image(rgb_data, output_rgb_path)
    normalize_and_save_image(ocv_data, output_ocv_path)

    print(f"Images saved: {output_rgb_path}, {output_ocv_path}")

if __name__ == "__main__":
    # Example usage
    input_image_path = "canvas.tiff"
    output_rgb_image_path = "lime_rgb.png"
    output_ocv_image_path = "lime_ocv.png"
    
    main(input_image_path, output_rgb_image_path, output_ocv_image_path)
