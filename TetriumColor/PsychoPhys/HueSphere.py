import math
from PIL import Image
import numpy as np
from typing import Callable, List
import numpy.typing as npt
import os

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from PIL import Image, ImageDraw
from TetriumColor.ColorMath import Geometry
from TetriumColor.TetraColorPicker import BackgroundNoiseGenerator
from TetriumColor.Utils.CustomTypes import ColorSpaceTransform, PlateColor, TetraColor
import TetriumColor.ColorMath.GamutMath as GamutMath
from TetriumColor.ColorMath.Geometry import ConvertCubeUVToXYZ, ExportGeometryToObjFile, GenerateGeometryFromVertices
from TetriumColor.PsychoPhys.IshiharaPlate import IshiharaPlate
from TetriumColor.Visualization.cubeMapViz import SetUp3DPlot


def GetSphereGeometry(luminance, saturation, num_points: int, filename: str, color_space_transform):
    """
    GetSphereGeometry generates a sphere geometry with fibonacci sampling.

    Args:
        num_points (int): number of points to sample for the sphere
        filename (str): filename for the OBJ file
    """
    # Generate sphere vertices
    vshh = GamutMath.SampleHueManifold(luminance, saturation, 4, num_points)
    hering = GamutMath.ConvertVSHToHering(vshh)
    vertices, triangles, normals, _ = Geometry.GenerateGeometryFromVertices(hering[:, 1:])
    uv_coords = Geometry.CartesianToUV(vertices)
    print(len(uv_coords))
    remapped_vshh = GamutMath.RemapGamutPoints(vshh, color_space_transform,
                                               GamutMath.GenerateGamutLUT(vshh, color_space_transform))
    tetra_colors: List[TetraColor] = GamutMath.ConvertVSHtoTetraColor(remapped_vshh, color_space_transform)

    rgb_colors = np.array([color.RGB for color in tetra_colors])
    # Export geometry to OBJ file
    ExportGeometryToObjFile(vertices, triangles, normals, uv_coords, rgb_colors, filename)

    return vertices, triangles, normals, uv_coords, rgb_colors


def GetFibonacciSampledHueTexture(num_points: int, luminance: float, saturation: float, color_space_transform: ColorSpaceTransform,
                                  rgb_texture_filename: str, ocv_texture_filename: str):
    """GetHueSphereGeometryWithLineTexture generates a hue sphere geometry with fibonacci sampling and line texture

    Args:
        num_points (int): number of points to sample for the sphere
        luminance (float): luminance value
        saturation (float): saturation value
        color_space_transform (ColorSpaceTransform): color space transform object
        rgb_texture_filename (str): filename for the RGB texture
        ocv_texture_filename (str): filename for the OCV texture
    """
    # ------- TEXTURE CREATION
    # Split color tuples into RGB and OCV components
    # Make Texture First, to fill the entire texture
    image_size = np.sqrt(num_points)
    image_size = int(2 ** np.ceil(np.log2(image_size)))
    uv_coords = np.stack(np.meshgrid(np.linspace(0, 1, image_size, endpoint=False),
                         np.linspace(0, 1, image_size, endpoint=False)), -1).reshape(-1, 2)

    # Get corresponding color from the point
    cartesian = Geometry.UVToCartesian(uv_coords, saturation)
    # Add a new column to the cartesian array
    lum_column = np.ones((cartesian.shape[0], 1)) * luminance
    hering = np.hstack((lum_column, cartesian))

    vshh = GamutMath.ConvertHeringToVSH(hering)
    remapped_vshh = GamutMath.RemapGamutPoints(vshh, color_space_transform,
                                               GamutMath.GenerateGamutLUT(vshh, color_space_transform))

    tetra_colors: List[TetraColor] = GamutMath.ConvertVSHtoTetraColor(remapped_vshh, color_space_transform)
    rgb_colors = np.array([color.RGB for color in tetra_colors])

    vertices, triangles, normals, indices = Geometry.GenerateGeometryFromVertices(hering[:, 1:])
    ExportGeometryToObjFile(vertices, triangles, normals,
                            uv_coords[indices], rgb_colors, './tmp/geometry/sphere_debugging.obj')

    # sample color per pixel to avoid empty spots
    image_rgb = Image.new('RGB', (image_size, image_size))
    image_ocv = Image.new('RGB', (image_size, image_size))

    draw_rgb = ImageDraw.Draw(image_rgb)
    draw_ocv = ImageDraw.Draw(image_ocv)

    for color, (u, v) in zip(tetra_colors, uv_coords):
        image_location = (int(v * image_size), int(u * image_size))
        rgb_color = (int(color.RGB[0] * 255), int(color.RGB[1] * 255), int(color.RGB[2] * 255))
        draw_rgb.point(image_location, fill=rgb_color)
        ocv_color = (int(color.OCV[0] * 255), int(color.OCV[1] * 255), int(color.OCV[2] * 255))
        draw_ocv.point(image_location, fill=ocv_color)
    # Save textures as an image
    image_rgb.save(rgb_texture_filename)
    image_ocv.save(ocv_texture_filename)
    return uv_coords, tetra_colors, cartesian


def GenerateCubeMapTextures(luminance: float, saturation: float, color_space_transform: ColorSpaceTransform,
                            image_size: int, filename_RGB: str, filename_OCV: str):
    """GenerateCubeMapTextures generates the cube map textures for a given luminance and saturation.

    Args:
        luminance (float): luminance value
        saturation (float): saturation value
        color_space_transform (ColorSpaceTransform): color space transform object
        filename_RGB (str): filename for the RGB cube map texture
        filename_OCV (str): filename for the OpenCV cube map texture
    Returns:
        Saves the generated textures to the specified filenames.
    """
    # Grid of UV coordinate that are the size of image_size
    all_us = (np.arange(image_size) + 0.5) / image_size
    all_vs = (np.arange(image_size) + 0.5) / image_size
    cube_u, cube_v = np.meshgrid(all_us, all_vs)
    flattened_u, flattened_v = cube_u.flatten(), cube_v.flatten()

    # change the associated xyzs -> to a new direction, but the same color values
    qDirMat = GamutMath.GetTransformChromToQDir(color_space_transform)
    invQDirMat = np.linalg.inv(qDirMat)

    # Create the RGB/OCV GenerateCubeMapTextures
    for i in range(6):
        img_rgb = Image.new('RGB', (image_size, image_size))  # sample color per pixel to avoid empty spots
        img_ocv = Image.new('RGB', (image_size, image_size))

        draw_rgb = ImageDraw.Draw(img_rgb)
        draw_ocv = ImageDraw.Draw(img_ocv)
        # ax = SetUp3DPlot()

        # convert the xyz coordinates of the cube map back into the original hering space -- this defines the
        # cubemap directions exactly !
        xyz = ConvertCubeUVToXYZ(i, cube_u, cube_v, saturation).reshape(-1, 3)
        # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='b', marker='o', label='Vertices')

        xyz = np.dot(invQDirMat, xyz.T).T
        lum_vector = luminance * np.ones(image_size * image_size)

        vxyz = np.hstack((lum_vector[np.newaxis, :].T, xyz))
        vshh = GamutMath.ConvertHeringToVSH(vxyz)

        vxyz_back = GamutMath.ConvertVSHToHering(vshh)
        # ax.scatter(vxyz_back[:, 1], vxyz_back[:, 2], vxyz_back[:, 3], c='r', marker='^', label='Tetra Cartesian')
        # plt.show()

        map_angle_sat = GamutMath.GenerateGamutLUT(vshh, color_space_transform)
        remapped_vshh = GamutMath.RemapGamutPoints(vshh, color_space_transform, map_angle_sat)
        corresponding_tetracolors = GamutMath.ConvertVSHtoTetraColor(remapped_vshh, color_space_transform)

        for j in range(len(flattened_u)):
            u, v = flattened_v[j], flattened_u[j]  # swap axis for PIL
            color: TetraColor = corresponding_tetracolors[j]
            rgb_color = (int(color.RGB[0] * 255), int(color.RGB[1] * 255), int(color.RGB[2] * 255))
            draw_rgb.point((u * image_size, v * image_size), fill=rgb_color)
            ocv_color = (int(color.OCV[0] * 255), int(color.OCV[1] * 255), int(color.OCV[2] * 255))
            draw_ocv.point((u * image_size, v * image_size), fill=ocv_color)

        # Save the images
        img_rgb.save(f'{filename_RGB}_{str(i)}.png')
        img_ocv.save(f'{filename_OCV}_{str(i)}.png')


def ConcatenateCubeMap(basename: str, output_filename: str):
    """
    Concatenate cubemap textures into a single cross-layout image with correct orientation.

    Parameters:
        basename (str): The base name of the input files, e.g., "texture". Files are assumed to follow the format "<basename>_i.png".
                       `i` corresponds to the index: 0 (+X), 1 (-X), 2 (+Y), 3 (-Y), 4 (+Z), 5 (-Z).
        output_filename (str): The output file name for the concatenated image.
    """
    # Load images for each face
    faces = []
    for i in range(6):
        filename = f"{basename}_{i}.png"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing cubemap texture: {filename}")
        faces.append(Image.open(filename))

    # Assume all faces are the same size
    face_width, face_height = faces[0].size

    # Create a blank image for the cross layout
    width = 4 * face_width
    height = 3 * face_height
    cubemap_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # +X (0)
    cubemap_image.paste(faces[0], (2 * face_width, face_height))
    # -X (1) flipped horizontally
    cubemap_image.paste(faces[1], (0, face_height))
    # +Y (2) flipped vertically
    cubemap_image.paste(faces[3], (face_width, 0))  # swap 2 and 3 because of the flipped orientation i think
    # -Y (3) flipped vertically
    cubemap_image.paste(faces[2], (face_width, 2 * face_height))
    # +Z (4)
    cubemap_image.paste(faces[4], (face_width, face_height))
    # -Z (5) flipped horizontally
    cubemap_image.paste(faces[5], (3 * face_width, face_height))

    # Save the concatenated image
    cubemap_image.save(output_filename)
    print(f"Saved concatenated cubemap to {output_filename}")


def CreateCircleGrid(grid: npt.NDArray, padding: int, radius: int, output_base: str):
    """
    Creates two images from grids of colors, where each grid cell is represented by a circle.

    Args:
        grid (np.ndarray): First grid of colors with shape (s, s, 2, dim).
        padding (int): Padding around the grid in pixels.
        radius (int): Radius of each circle in pixels.
        output_base (str): Base filename for the output images, consisting of RGB OCV pairs.
    """
    def __createPairImages(metamer_idx):
        s = grid.shape[0]  # Grid size (s x s)

        # Image size calculation
        cell_size = 2 * radius  # Each cell is defined by the diameter of the circle
        grid_size = s * cell_size + (s + 1) * padding
        image_size = (grid_size, grid_size)

        # Create blank image
        img_RGB = Image.new("RGB", image_size, "black")
        img_OCV = Image.new("RGB", image_size, "black")

        draw_RGB = ImageDraw.Draw(img_RGB)
        draw_OCV = ImageDraw.Draw(img_OCV)

        # Loop over the grid
        for i in range(s):
            for j in range(s):
                # Compute circle center
                cx = padding + j * (cell_size + padding) + radius
                cy = padding + i * (cell_size + padding) + radius

                # Convert color to tuple and scale (assuming colors are normalized [0, 1])
                color = tuple((grid[i, j, metamer_idx, :3] * 255).astype(int))  # Use the first color in the pair
                draw_RGB.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline="black")

                color = tuple((grid[i, j, metamer_idx, 3:] * 255).astype(int))  # Use the first color in the pair
                draw_OCV.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline="black")

            # Save image
        img_RGB.save(output_base + f'_{metamer_idx}_RGB.png')
        img_OCV.save(output_base + f'_{metamer_idx}_OCV.png')

    __createPairImages(0)
    __createPairImages(1)


def CreatePaddedGrid(image_files, grid_size=None, padding=10, bg_color=(0, 0, 0)):
    """
    Create a padded grid of images from a list of image files.

    Args:
        image_files (list of str): List of image file paths.
        grid_size (tuple, optional): Tuple (rows, cols) specifying grid dimensions.
                                     If None, grid is square.
        padding (int, optional): Padding between images in pixels. Defaults to 10.
        bg_color (tuple, optional): Background color for the grid (R, G, B). Defaults to white.

    Returns:
        Image: The grid as a Pillow Image object.
    """
    # Load all images
    images = [Image.open(file) for file in image_files]

    # Ensure all images are the same size
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    resized_images = [img.resize((max_width, max_height)) for img in images]

    # Determine grid size if not provided
    num_images = len(images)
    if grid_size is None:
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
    else:
        rows, cols = grid_size

    # Calculate final grid dimensions
    grid_width = cols * max_width + (cols - 1) * padding
    grid_height = rows * max_height + (rows - 1) * padding

    # Create a blank canvas for the grid
    grid_image = Image.new("RGB", (grid_width, grid_height), bg_color)

    # Paste images onto the grid
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = col * (max_width + padding)
        y = row * (max_height + padding)
        grid_image.paste(img, (x, y))

    return grid_image


def CreatePseudoIsochromaticGrid(grid, output_dir: str, output_base: str, seed: int = 42, noise_generator: BackgroundNoiseGenerator | None = None):
    subdirname = f"./{output_dir}/sub_images"
    os.makedirs(subdirname, exist_ok=True)
    plate: IshiharaPlate = IshiharaPlate(seed=seed)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            metamer1 = TetraColor(grid[i, j, 0, :3], grid[i, j, 0, 3:])
            metamer2 = TetraColor(grid[i, j, 1, :3], grid[i, j, 1, 3:])
            plate_color = PlateColor(metamer1, metamer2)
            noise_generator_fn = noise_generator.GenerateNoiseFunction(plate_color) if noise_generator else None
            plate.GeneratePlate(seed, -1, plate_color, noise_generator_fn)
            plate.ExportPlate(os.path.join(subdirname, f"{output_base}_{i}_{j}_RGB.png"),
                              os.path.join(subdirname, f"{output_base}_{i}_{j}_OCV.png"))

    img_rgb = CreatePaddedGrid([os.path.join(subdirname, f"{output_base}_{i}_{j}_RGB.png") for i in range(grid.shape[0])
                                for j in range(grid.shape[1])])
    img_rgb = img_rgb.resize((1024, 1024), Image.Resampling.BOX)
    img_rgb.save(f"./{output_dir}/{output_base}_RGB.png")

    img_ocv = CreatePaddedGrid([os.path.join(subdirname, f"{output_base}_{i}_{j}_OCV.png") for i in range(grid.shape[0])
                                for j in range(grid.shape[1])])
    img_ocv = img_ocv.resize((1024, 1024), Image.Resampling.BOX)
    img_ocv.save(f"./{output_dir}/{output_base}_OCV.png")


def CreatePseudoIsochromaticImages(colors, output_dir: str, output_base: str, seed=42):
    subdirname = f"./{output_dir}/sub_images"
    os.makedirs(subdirname, exist_ok=True)
    plate: IshiharaPlate = IshiharaPlate(seed=seed)
    for i in range(len(colors)):
        metamer1 = TetraColor(colors[i, 0, :3], colors[i, 0, 3:])
        metamer2 = TetraColor(colors[i, 1, :3], colors[i, 1, 3:])
        plate_color = PlateColor(metamer1, metamer2)
        plate.GeneratePlate(seed, -1, plate_color)
        plate.ExportPlate(os.path.join(subdirname, f"{output_base}_{i:03}_RGB.png"),
                          os.path.join(subdirname, f"{output_base}_{i:03}_OCV.png"))
