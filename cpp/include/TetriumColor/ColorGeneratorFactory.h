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
    // Create a GeneticColorGenerator instance
    static PyObject* CreateGeneticColorGenerator(
        const std::string& sex = "female",
        float percentage_screened = 0.999f,
        float peak_to_test = 547.0f,
        float luminance = 1.0f,
        float saturation = 0.5f,
        const std::vector<int>& dimensions = {2},
        int seed = 42,
        int trials_per_direction = 20,
        const std::vector<int>& metameric_axes = {}, // Empty = default [1, 2, 3]
        const std::string& display_primaries_path = ""
    );

    // Create a QuestColorGenerator instance
    static PyObject* CreateQuestColorGenerator(
        const std::string& sex = "female",
        float percentage_screened = 0.999f,
        float background_luminance = 0.5f,
        int trials_per_direction = 20,
        const std::vector<int>& metameric_axes = {}, // Empty = all axes
        const std::vector<int>& dimensions = {2},    // Dimensions for ObserverGenotypes
        const std::string& display_primaries_path = ""
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
};

} // namespace TetriumColor
