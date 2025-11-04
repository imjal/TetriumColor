from TetriumColor import ColorSpace, ColorSampler, ColorSpaceType
import TetriumColor.Visualization as viz
from TetriumColor.Utils.ParserOptions import AddAnimationArgs, AddObserverArgs, AddVideoOutputArgs
from TetriumColor.Utils.ImageUtils import CreatePaddedGrid
from TetriumColor.Measurement import load_primaries_from_csv, compare_dataset_to_primaries, get_spectras_from_rgbo_list, export_metamer_difference, export_predicted_vs_measured_with_square_coords, get_spectras_from_rgbo_list, plot_measured_vs_predicted
from TetriumColor.Observer import Observer, Spectra
import pdb
from PIL import Image
import numpy as np
import numpy.typing as npt
import argparse

from typing import List

import tetrapolyscope as ps
import matplotlib.pyplot as plt


def renormalize_spectra(spectras: List[Spectra], observer, primaries: List[Spectra], scaling_factor: float = 10000) -> npt.NDArray:

    disp = observer.observe_spectras(primaries)  # each row is a cone_vec
    intensities = disp.T * scaling_factor  # each column is a cone_vec
    white_pt = observer.observe_normalized(np.ones_like(observer.wavelengths))
    white_weights = np.linalg.inv(intensities)@white_pt
    return white_weights


# Load Observer and Measured Primaries
parser = argparse.ArgumentParser(description='Visualize Cones from Tetra Observers')
AddObserverArgs(parser)
AddAnimationArgs(parser)
AddVideoOutputArgs(parser)
parser.add_argument('--scrambleProb', type=float, default=0, help='Probability of scrambling the color')
args = parser.parse_args()


# Load Observer and Measured Primaries
wavelengths = np.arange(360, 831, 1)
observer = Observer.custom_observer(wavelengths, args.od, args.dimension, args.s_cone_peak, args.m_cone_peak, args.q_cone_peak,
                                    args.l_cone_peak, args.macula, args.lens, args.template)
primaries: List[Spectra] = load_primaries_from_csv("../../measurements/2025-05-21/primaries")
# primaries = [primaries[i] for i in [2, 1, 0, 3]]  # ORDER primaries as RGBO

metameric_axis = 2

# colors = ['red', 'green', 'blue', 'black']
# for i,  primary in enumerate(primaries):
#     plt.plot(primary.wavelengths, primary.data, color=colors[i])

# plt.show()

cs_4d = ColorSpace(observer,
                   display_primaries=primaries, metameric_axis=metameric_axis)

color_sampler = ColorSampler(cs_4d, cubemap_size=5)

images = color_sampler.generate_cubemap(1.0, 0.4, ColorSpaceType.SRGB)
# for img in images[4:]:
#     img.save("./results/cubemap_" + str(images.index(img)) + ".png")
image = color_sampler._concatenate_cubemap(images)
# Apply gamma encoding to the image
gamma = 2.2
gamma_corrected_image = np.clip((np.array(image) / 255.0) ** (1 / gamma) * 255, 0, 255).astype(np.uint8)
image = Image.fromarray(gamma_corrected_image)
image.save("./results/cubemap.png")

colors = color_sampler.output_cubemap_values(1.0, 0.4, ColorSpaceType.DISP)[4:]
# images = color_sampler.get_metameric_grid_plates(1.0, 0.4, 4)
# img = CreatePaddedGrid([i[0] for i in images], padding=0, canvas_size=(1280 * 8, 720 * 8))
# img.save("all.png")
# img_rgo = CreatePaddedGrid([i[0] for i in images], padding=0)
# img_rgo.save("newgrid_RGB.png")
# img_bgo = CreatePaddedGrid([i[1] for i in images], padding=0)
# img_bgo.save("newgrid_OCV.png")

# idxs = [(2, 3), (2, 4), (3, 3), (3, 4)]
# idxs = [j * 5 + i for i, j in idxs]
# for i, j in idxs:
#     images[i * 5 + j][0].save("./results/cubemap_" + str(i * 5 + j) + ".png")

# colors, cones = color_sampler.get_metameric_pairs(1.0, 0.4, 4)
# print(cones)
colors = np.array([item for sublist in colors for item in sublist])
# idxs = [6, 7, 9, 23, 40, 34, 35, 36, 40]
# colors = colors[idxs]
colors_8bit = (np.array(colors) * 255).astype(np.uint8)
print(repr(colors_8bit))
exit()
print("{" + ",\n ".join("{" + ", ".join(map(str, inner_list)) + "}" for inner_list in colors_8bit) + "}")


measurements_dir = "../../measurements/2025-05-21/5x5-cubemap/"
measured_spectras = get_spectras_from_rgbo_list(measurements_dir, colors_8bit.tolist())
# results = compare_dataset_to_primaries(measurements_dir, colors_8bit.tolist(), primaries)
export_predicted_vs_measured_with_square_coords(
    measurements_dir, colors_8bit.tolist(), primaries, "./results/")
# export_metamer_difference(observer, cs_4d, measurements_dir, colors_8bit.tolist(), primaries, "./results/")


# Polyscope Animation Inits
ps.init()
ps.set_always_redraw(True)
ps.set_ground_plane_mode('shadow_only')
ps.set_SSAA_factor(2)
ps.set_window_size(720, 720)


cs_for_viz = ColorSpace(observer)

Q_cone = np.array([[0, 0, 0, 0], [0, 0, 1, 0]])
line = cs_for_viz.convert(Q_cone, ColorSpaceType.CONE, ColorSpaceType.HERING)[:, 1:]
line[1] = line[1]/np.linalg.norm(line[1])
line[0] = -line[1]
viz.Render3DLine("line", line, np.zeros(3), 0.005)

# viz.RenderOBS("observer", cs_for_viz, args.display_basis)
# ps.get_surface_mesh("observer").set_transparency(0.5)
# viz.AnimationUtils.AddObject("observer", "surface_mesh",
#                              args.position, args.velocity, args.rotation_axis, args.rotation_speed)

srgb_vals = cs_4d.convert(colors, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.SRGB)
hering_vals = cs_4d.convert(colors, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.HERING)[:, 1:]
real_cone_vals = cs_4d.convert(colors, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.CONE)
viz.RenderPointCloud("cube-map-predicted", hering_vals, srgb_vals)
maxbasis_vals_intended = cs_4d.convert(colors, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.MAXBASIS)


# primary_spectra = sum(scale * primary.data for scale, primary in zip(colors_8bit[0] / 255.0, primaries))
# measured_spectras = [Spectra(wavelengths=primaries[0].wavelengths, data=primary_spectra)]
# primary_spectra2 = sum(scale * primary.data for scale, primary in zip(colors_8bit[1] / 255.0, primaries))
# measured_spectras.append(Spectra(wavelengths=primaries[0].wavelengths, data=primary_spectra2))

measured = observer.observe_spectras(measured_spectras) * 10000
white_weights = renormalize_spectra(measured_spectras, observer, primaries)
disp_vals = cs_4d.convert(measured, from_space=ColorSpaceType.CONE, to_space=ColorSpaceType.DISP) * white_weights
hering_vals_new = cs_4d.convert(disp_vals, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.HERING)[:, 1:]
sRGBvals_new = cs_4d.convert(disp_vals, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.SRGB)
cone_vals = cs_4d.convert(disp_vals, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.CONE)
maxbasis_vals = cs_4d.convert(disp_vals, from_space=ColorSpaceType.DISP, to_space=ColorSpaceType.MAXBASIS)

print(np.mean(real_cone_vals - cone_vals)**2)
mean_squared_error = np.mean(hering_vals - hering_vals_new)**2
print(mean_squared_error)


arr = np.zeros((5, 5, 2, 4))
real_arr = np.zeros((5, 5, 2, 4))
for k in range(2):

    # Render a 5x5 cube image of all colors in sRGBvals_new
    cube_image = np.zeros((5, 5, 3), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if k == 1:
                i = 4 - i
            cube_image[j, i] = (sRGBvals_new[k * 25 + idx] * 255).astype(np.uint8)
            # cube_image[j, i] = (sRGBvals_new[idx] * 255).astype(np.uint8)
            arr[j, i, k] = maxbasis_vals[k * 25 + idx]
            real_arr[j, i, k] = maxbasis_vals_intended[k * 25 + idx]

    plt.imshow(cube_image)
    plt.axis('off')
    plt.savefig(f"./results/{k}_cube_image.png")
    plt.show()

print(arr)
print(real_arr)

pdb.set_trace()

viz.RenderPointCloud("cube-map-spectras", hering_vals_new, sRGBvals_new)


# Need to call this after registering structures
ps.set_automatically_compute_scene_extents(False)
# Output Video to Screen or Save to File (based on options)
if args.output_filename:
    fd = viz.OpenVideo(args.output_filename)
    viz.RenderVideo(fd, args.total_frames, args.fps)
    viz.CloseVideo(fd)
else:
    delta_time: float = 1 / args.fps

    def callback():
        pass
        # viz.AnimationUtils.UpdateObjects(delta_time)
    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()
