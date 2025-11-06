#pragma once

#include "ColorSpaceType.h"
#include "QuestColorPicker.h"
#include <string>

namespace TetriumColor
{
class QuestColorPickerPlateGenerator
{
  public:
    QuestColorPickerPlateGenerator(QuestColorPicker& color_picker, int seed = 42);

    ~QuestColorPickerPlateGenerator();

    // Generate a plate for a specific direction index
    void GetPlate(
        int direction_idx,
        const std::string& filename,
        int hidden_number,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

    void GetPlate(
        int direction_idx,
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
    QuestColorPicker& colorPicker;
    int seed;
};
} // namespace TetriumColor
