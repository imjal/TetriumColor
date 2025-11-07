#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace TetriumColor
{

// Trial data for pseudo-isochromatic plate tests
struct PseudoIsochromaticTrial
{
    std::string genotype;
    int metameric_axis;
    std::string rgb_path;
    std::string ocv_path;
    std::string hidden_symbol;
    double intensity;

    // Metadata for logging
    std::map<std::string, std::string> metadata;
};

// Trial data for circle grid (scrambled face) tests
struct CircleGridTrial
{
    std::string genotype;
    int metameric_axis;
    std::vector<std::string> image_paths;
    std::vector<int> scramble_indices;

    // Metadata for logging
    std::map<std::string, std::string> metadata;
};

// Variant to hold different trial types
using TrialData = std::variant<PseudoIsochromaticTrial, CircleGridTrial>;

} // namespace TetriumColor
