#pragma once

#include "ColorGenerator.h"
#include "ColorSpaceType.h"
#include "ColorTestResult.h"
#include "GeneticColorPicker.h"
#include "GeneticColorPickerPlateGenerator.h"
#include "PseudoIsochromaticPlateGenerator.h"
#include "SolidColorGenerator.h"

#include <string>
#include <vector>

namespace TetriumColor
{
void Init();
void Cleanup();

class CircleGridGenerator
{
  public:
    CircleGridGenerator(
        float scramble_prob,
        const std::string& sex,
        float percentage_screened,
        float peak_to_test = 547.0f,
        float luminance = 1.0f,
        float saturation = 0.5f,
        const std::vector<int>& dimensions = {2},
        int seed = 42,
        const std::string& cst_display_type = "led",
        const std::string& display_primaries_path = ""
    );
    ~CircleGridGenerator();

    // Get the list of genotypes
    std::vector<std::string> GetGenotypes() const;

    // Returns index list for correct/scramble mapping; writes images to filenames with suffixes
    // For SRGB: writes filenames[i] + ".png"
    // For DISP_6P: writes filenames[i] + "_RGB.png" and "_OCV.png"
    std::vector<std::pair<int, int>> GetImages(
        const std::string& genotype,
        int metameric_axis,
        const std::vector<std::string>& filenames,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P
    );

  private:
    void* pModule = nullptr;
    void* pClass = nullptr;
    void* pInstance = nullptr;
};
} // namespace TetriumColor
