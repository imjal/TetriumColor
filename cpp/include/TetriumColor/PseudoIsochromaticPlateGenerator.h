#pragma once

#include <string>
#include "ColorTestResult.h"
#include "ColorSpaceType.h"
#include "ColorGenerator.h"

namespace TetriumColor
{
class PseudoIsochromaticPlateGenerator
{
  public:
    PseudoIsochromaticPlateGenerator(
        ColorGenerator& color_generator,
        int seed = 42
    );

    ~PseudoIsochromaticPlateGenerator();

    void NewPlate(
        const std::string& filename,
        int hidden_number,
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

  private:
    void* pModule;
    void* pClass;
    void* pInstance;
};
} // namespace TetriumColor
