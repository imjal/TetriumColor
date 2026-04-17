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
    // Create a GeneticColorGenerator instance (Method of Constant Stimuli)
    static PyObject* CreateGeneticColorGenerator(
        const std::string& sex = "female",
        float percentage_screened = 0.999f,
        float peak_to_test = 547.0f,
        float luminance = 0.5f,
        float saturation = 0.5f,
        const std::vector<int>& dimensions = {2},
        int seed = 42,
        int trials_per_direction = 50,
        const std::vector<int>& metameric_axes = {}, // Empty = default [1, 2, 3]
        const std::string& display_primaries_path = "",
        float degree = 4.0f,
        int mcs_k = 1,      // number of MCS intensity levels (1 = max metamer only)
        bool debug_middle = false // always pick the center cubemap point (2,2) instead of random
    );

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
        float degree = 4.0f
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
};

} // namespace TetriumColor
