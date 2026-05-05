#pragma once

#include <Python.h>
#include <string>
#include <vector>

namespace TetriumColor
{

// Factory for creating Python ColorGenerator instances
class ColorGeneratorFactory
{
  public:
    // Create a QuestColorGenerator instance
    static PyObject* CreateQuestColorGenerator(
        const std::string& sex = "female",
        float percentage_screened = 0.999f,
        float background_luminance = 0.5f,
        int trials_per_direction = 20,
        const std::vector<int>& metameric_axes = {}, // Empty = all axes
        const std::vector<int>& dimensions = {2},    // Dimensions for ObserverGenotypes
        const std::string& display_primaries_path = "",
        bool bipolar = false,
        float degree = 4.0f,
        int mcs_k = 0, // >0: use MCS with K equally-spaced levels instead of Quest adaptive
        const std::vector<int>& observer_indices = {},
        const std::string& color_picking_space = "cone_contrast"
    );

    // Create an AEPsychThresholdContourGenerator instance
    static PyObject* CreateAEPsychThresholdContourGenerator(
        int n_trials = 300,
        int n_sobol = 20,
        float threshold_level = 0.75f,
        const std::string& sex = "both",
        float background_luminance = 0.5f,
        const std::vector<int>& dimensions = {3},
        const std::string& display_primaries_path = "",
        int seed = 42,
        int n_cmf_samples = 200,
        float patch_sigma_scale = 5.0f,
        float min_patch_major = 0.15f,
        float min_patch_minor = 0.06f,
        float max_radius = 0.65f
    );

    // Create a TestGenerator (PseudoIsochromaticPlateGenerator) instance
    static PyObject* CreatePseudoIsochromaticPlateGenerator(
        PyObject* color_generator,
        int seed = 42
    );

    // Create a TestGenerator (CircleGridGenerator) instance
    static PyObject* CreateCircleGridGenerator(
        PyObject* color_generator,
        float scramble_prob = 0.5f,
        float luminance = 1.0f,
        float saturation = 0.5f
    );

    // Create a TestGenerator (BipartiteFieldGenerator) instance
    static PyObject* CreateBipartiteFieldGenerator(
        PyObject* color_generator,
        int seed = 42,
        int size = 512
    );

    // Create a TestGenerator (GaussianBlobGenerator) instance
    static PyObject* CreateGaussianBlobGenerator(
        PyObject* color_generator,
        int seed = 42,
        int size = 1024,
        float blob_size = 1.0f,
        bool constant_disp_background = false
    );

    // Get observer CDF as a vector of cumulative probabilities (one per observer,
    // sorted by decreasing individual probability). Index N-1 gives the population
    // coverage when testing the N most common observers.
    static std::vector<float> GetObserverCDF(const std::string& sex, int dimension);
};

} // namespace TetriumColor
