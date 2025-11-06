#pragma once

#include <map>
#include <string>
#include <vector>

namespace TetriumColor
{
class QuestColorPicker
{
  public:
    QuestColorPicker(
        const std::string& mode = "cone_shift",
        int num_genotypes = 8,
        int trials_per_direction = 20,
        const std::string& sex = "both",
        float background_luminance = 0.5f,
        int seed = 42,
        const std::string& display_primaries_path = "",
        const std::vector<int>& metameric_axes = {} // empty = all axes
    );

    ~QuestColorPicker();

    // Get directions metadata (for generating plates)
    // Returns a map of direction index to genotype and metameric axis info
    std::map<int, std::pair<std::string, int>> GetDirectionsMetadata() const;

    // Get the number of directions
    size_t GetNumDirections() const;

    // Export thresholds to CSV
    void ExportThresholds(const std::string& filename);

    // Get the Python instance (for internal use by other C++ wrappers)
    void* GetPythonInstance() const { return pInstance; }

  private:
    void* pModule;
    void* pClass;
    void* pInstance;
};
} // namespace TetriumColor
