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
        int mcs_k = 0  // >0: use MCS with K equally-spaced levels instead of Quest adaptive
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
        int size = 1024
    );

    // Get observer CDF as a vector of cumulative probabilities (one per observer,
    // sorted by decreasing individual probability). Index N-1 gives the population
    // coverage when testing the N most common observers.
    static std::vector<float> GetObserverCDF(const std::string& sex, int dimension);
};

} // namespace TetriumColor
