#pragma once

#include "ColorSpaceType.h"
#include <string>
#include <utility>

namespace TetriumColor
{

class SolidColorGenerator
{
  public:
    SolidColorGenerator(const std::string& primary_path);
    ~SolidColorGenerator();

    // Generate a solid color circle and return paths to RGB and OCV images
    // rgbo_values: RGBO color values (0-255 range)
    // Returns: pair of (rgb_path, ocv_path)
    std::pair<std::string, std::string> GenerateCircle(
        const std::string& filename_base,
        float r,
        float g,
        float b,
        float o,
        int image_size = 512,
        float circle_radius_ratio = 0.8f,
        bool has_noisy_boundary = false,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P
    );

  private:
    void* pModule = nullptr;
    void* pClass = nullptr;
    void* pInstance = nullptr;
};

} // namespace TetriumColor
