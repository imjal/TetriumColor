#pragma once

#include <string>
#include <vector>

namespace TetriumColor
{
class ColorGenerator
{
  public:
    ColorGenerator(
        const std::string& sex,
        float percentage_screened,
        float peak_to_test,
        const std::vector<int>& dimensions,
        const std::string& cst_display_type,
        const std::string& display_primaries_path
    );

    ~ColorGenerator();

    int GetNumSamples();

  private:
    void* pModule;
    void* pClass;
    void* pInstance;

    friend class PseudoIsochromaticPlateGenerator;
};
} // namespace TetriumColor
