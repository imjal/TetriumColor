#pragma once

#include "ColorGenerator.h"
#include "ColorSpaceType.h"
#include "ColorTestResult.h"
#include <string>

namespace TetriumColor
{
class PseudoIsochromaticPlateGenerator
{
  public:
    PseudoIsochromaticPlateGenerator(ColorGenerator& color_generator, int seed = 42);

    ~PseudoIsochromaticPlateGenerator();

    void NewPlate(
        const std::string& filename,
        int hidden_number,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

    void NewPlate(
        const std::string& filename,
        const std::string& hidden_symbol,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

    void GetPlate(
        ColorTestResult previous_result,
        const std::string& filename,
        int hidden_number,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

    void GetPlate(
        ColorTestResult previous_result,
        const std::string& filename,
        const std::string& hidden_symbol,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

    void GetLuminancePlate(
        const std::string& filename,
        const std::string& hidden_symbol,
        ColorSpaceType output_space = ColorSpaceType::SRGB,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

  private:
    void* pModule;
    void* pClass;
    void* pInstance;
    void* pDefaultColorSpace;
};
} // namespace TetriumColor
