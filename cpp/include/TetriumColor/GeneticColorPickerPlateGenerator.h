#pragma once

#include "ColorSpaceType.h"
#include "GeneticColorPicker.h"
#include <string>

namespace TetriumColor
{
class GeneticColorPickerPlateGenerator
{
  public:
    GeneticColorPickerPlateGenerator(GeneticColorPicker& color_picker, int seed = 42);

    ~GeneticColorPickerPlateGenerator();

    // Generate a plate for a specific genotype and metameric axis
    void GetPlate(
        const std::string& genotype,
        int metameric_axis,
        const std::string& filename,
        int hidden_number,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

    void GetPlate(
        const std::string& genotype,
        int metameric_axis,
        const std::string& filename,
        const std::string& hidden_symbol,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

  private:
    void* pModule;
    void* pClass;
    void* pInstance;
    GeneticColorPicker& colorPicker;
};
} // namespace TetriumColor
