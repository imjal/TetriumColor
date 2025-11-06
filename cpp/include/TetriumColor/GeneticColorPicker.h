#pragma once

#include <string>
#include <vector>

namespace TetriumColor
{
class GeneticColorPicker
{
  public:
    GeneticColorPicker(
        const std::string& sex,
        float percentage_screened,
        float peak_to_test = 547.0f,
        float luminance = 1.0f,
        float saturation = 0.5f,
        const std::vector<int>& dimensions = {1, 2},
        int seed = 42,
        const std::string& display_primaries_path = ""
    );

    ~GeneticColorPicker();

    // Get the list of genotypes
    std::vector<std::string> GetGenotypes() const;

    // Get the number of genotypes
    size_t GetNumGenotypes() const;

    // Get the Python instance (for internal use by other C++ wrappers)
    void* GetPythonInstance() const { return pInstance; }

  private:
    void* pModule;
    void* pClass;
    void* pInstance;
};
} // namespace TetriumColor
