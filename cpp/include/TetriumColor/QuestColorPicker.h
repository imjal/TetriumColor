#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace TetriumColor
{

// Result from Quest GetColor
struct QuestColorResult
{
    int direction_idx;
    std::string genotype;
    int metameric_axis;
    double intensity; // Quest's recommended intensity [0, 1]
    bool is_done;     // true if all trials complete
};

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

    // Get the first trial (calls Python NewColor())
    QuestColorResult NewColor();

    // Get next trial and update Quest with previous response (calls Python GetColor())
    // correct: true if observer correctly identified the previous stimulus
    QuestColorResult GetColor(bool correct);

    // Export thresholds to CSV
    void ExportThresholds(const std::string& filename);

    // Get the Python instance (for internal use by other C++ wrappers)
    void* GetPythonInstance() const { return pInstance; }

  private:
    void* pModule;
    void* pClass;
    void* pInstance;

    int current_direction_idx = -1;
};
} // namespace TetriumColor
