#pragma once

#include "ColorSpaceType.h"
#include "ColorTestResult.h"
#include "TrialData.h"
#include <Python.h>
#include <optional>
#include <string>

namespace TetriumColor
{

// Generic wrapper for any Python TestGenerator instance
class TestGenerator
{
  public:
    // Constructor takes a Python TestGenerator instance
    TestGenerator(PyObject* python_test_generator);

    ~TestGenerator();

    // Get first trial (calls Python NewTest())
    std::optional<TrialData> NewTrial(
        const std::string& filename,
        const std::string& hidden_symbol = "",
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f,
        const std::string& genotype = "", // Optional: genotype string like "(558.9, 530.3)"
        int metameric_axis = -1           // Optional: metameric axis index
    );

    // Get next trial based on previous result (calls Python GetTest())
    // Returns std::nullopt if test is complete (Python returns None)
    std::optional<TrialData> GetNextTrial(
        ColorTestResult previous_result,
        const std::string& filename,
        const std::string& hidden_symbol,
        ColorSpaceType output_space = ColorSpaceType::DISP_6P,
        float lum_noise = 0.0f,
        float s_cone_noise = 0.0f
    );

    // Get genotypes list (for CircleGridGenerator and similar)
    // Returns empty vector if method doesn't exist
    std::vector<std::string> GetGenotypes();

    // Get total number of trials (if available from color generator)
    // Returns -1 if not available
    int GetTotalTrials();

    // Get thresholds dictionary (for QuestColorGenerator)
    // Returns empty map if not available or not Quest
    std::map<int, std::map<std::string, std::string>> GetThresholds();

    // Export thresholds to CSV file (for QuestColorGenerator)
    // Returns true if successful, false otherwise
    bool ExportThresholds(const std::string& filename);

  private:
    PyObject* pInstance; // Python TestGenerator instance

    // Helper to parse Python dict to TrialData
    TrialData ParseDictToTrialData(PyObject* dict);

    // Helper to get string from dict
    std::string GetStringFromDict(
        PyObject* dict,
        const char* key,
        const std::string& default_val = ""
    );

    // Helper to get int from dict
    int GetIntFromDict(PyObject* dict, const char* key, int default_val = -1);

    // Helper to get double from dict
    double GetDoubleFromDict(PyObject* dict, const char* key, double default_val = 0.0);

    // Helper to convert C++ ColorSpaceType enum to Python enum
    PyObject* ColorSpaceTypeToPython(ColorSpaceType space_type);
};

} // namespace TetriumColor
