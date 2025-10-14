#pragma once
#include "ColorSpaceType.h"
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
        const std::string& primary_path,
        int num_samples,
        float scramble_prob = 0.5f
    );
    ~CircleGridGenerator();

    // Returns index list for correct/scramble mapping; writes images to filenames with suffixes
    // For SRGB: writes filenames[i] + ".png"
    // For DISP_6P: writes filenames[i] + "_RGB.png" and "_OCV.png"
    std::vector<std::pair<int, int>> GetImages(
        float luminance,
        float saturation,
        const std::vector<std::string>& filenames,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P
    );

  private:
    void* pModule = nullptr;
    void* pClass = nullptr;
    void* pInstance = nullptr;
};
} // namespace TetriumColor
