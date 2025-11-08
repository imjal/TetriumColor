#include "TetriumColor/TestGenerator.h"
#include <iostream>
#include <map>
#include <stdexcept>

namespace TetriumColor
{

TestGenerator::TestGenerator(PyObject* python_test_generator) : pInstance(python_test_generator)
{
    if (pInstance) {
        Py_INCREF(pInstance);
    }
}

TestGenerator::~TestGenerator()
{
    if (pInstance) {
        Py_DECREF(pInstance);
        pInstance = nullptr;
    }
}

PyObject* TestGenerator::ColorSpaceTypeToPython(ColorSpaceType space_type)
{
    PyObject* pColorSpaceModule = PyImport_ImportModule("TetriumColor");
    if (!pColorSpaceModule) {
        PyErr_Print();
        return nullptr;
    }
    PyObject* pColorSpaceType = PyObject_GetAttrString(pColorSpaceModule, "ColorSpaceType");
    Py_DECREF(pColorSpaceModule);
    if (!pColorSpaceType) {
        PyErr_Print();
        return nullptr;
    }
    const char* type_name = nullptr;
    switch (space_type) {
    case ColorSpaceType::DISP_6P:
        type_name = "DISP_6P";
        break;
    case ColorSpaceType::SRGB:
        type_name = "SRGB";
        break;
    case ColorSpaceType::RGB:
        type_name = "SRGB"; // Map RGB to SRGB
        break;
    case ColorSpaceType::OCV:
        type_name = "DISP_6P"; // Map OCV to DISP_6P for compatibility
        break;
    default:
        type_name = "DISP_6P";
        break;
    }
    PyObject* pValue = PyObject_GetAttrString(pColorSpaceType, type_name);
    Py_DECREF(pColorSpaceType);
    return pValue;
}

std::optional<TrialData> TestGenerator::NewTrial(
    const std::string& filename,
    const std::string& hidden_symbol,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise,
    const std::string& genotype,
    int metameric_axis
)
{
    if (!pInstance) {
        throw std::runtime_error("TestGenerator: Python instance is null");
    }

    // Convert C++ ColorSpaceType enum to Python enum
    PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
    if (!pOutputSpace) {
        throw std::runtime_error("TestGenerator: Failed to convert output_space to Python enum");
    }

    // Call Python NewTest method using PyObject_Call with tuple and kwargs
    // This allows us to use keyword arguments for optional parameters
    PyObject* pMethod = PyObject_GetAttrString(pInstance, "NewTest");
    if (!pMethod) {
        Py_DECREF(pOutputSpace);
        PyErr_Print();
        throw std::runtime_error("TestGenerator: Failed to get NewTest method");
    }

    // Build positional arguments tuple: (filename, hidden_symbol)
    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(filename.c_str()));
    PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(hidden_symbol.c_str()));

    // Build keyword arguments dict: {output_space, lum_noise, s_cone_noise, genotype,
    // metameric_axis}
    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "output_space", pOutputSpace); // Steals reference
    PyDict_SetItemString(pKwargs, "lum_noise", PyFloat_FromDouble(lum_noise));
    PyDict_SetItemString(pKwargs, "s_cone_noise", PyFloat_FromDouble(s_cone_noise));

    // Add optional genotype and metameric_axis if provided
    if (!genotype.empty()) {
        // Parse genotype string (e.g., "(558.9, 530.3)") to Python tuple
        PyObject* pGenotype
            = PyRun_String(genotype.c_str(), Py_eval_input, PyDict_New(), PyDict_New());
        if (pGenotype) {
            PyDict_SetItemString(pKwargs, "genotype", pGenotype); // Steals reference
        } else {
            PyErr_Clear(); // Clear error if parsing fails
        }
    }
    if (metameric_axis >= 0) {
        PyDict_SetItemString(pKwargs, "metameric_axis", PyLong_FromLong(metameric_axis));
    }

    PyObject* pResult = PyObject_Call(pMethod, pArgs, pKwargs);

    Py_DECREF(pMethod);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    // Note: pOutputSpace reference was stolen by PyDict_SetItemString, don't DECREF it

    if (!pResult) {
        PyErr_Print();
        throw std::runtime_error("TestGenerator: Failed to call NewTest()");
    }

    // Parse the returned dict to TrialData
    TrialData trial_data = ParseDictToTrialData(pResult);
    Py_DECREF(pResult);

    return trial_data;
}

std::optional<TrialData> TestGenerator::GetNextTrial(
    ColorTestResult previous_result,
    const std::string& filename,
    const std::string& hidden_symbol,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise
)
{
    if (!pInstance) {
        throw std::runtime_error("TestGenerator: Python instance is null");
    }

    // Import ColorTestResult enum from Python
    PyObject* pModule = PyImport_ImportModule("TetriumColor.Utils.CustomTypes");
    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("TestGenerator: Failed to import CustomTypes module");
    }

    PyObject* pColorTestResultClass = PyObject_GetAttrString(pModule, "ColorTestResult");
    Py_DECREF(pModule);

    if (!pColorTestResultClass) {
        PyErr_Print();
        throw std::runtime_error("TestGenerator: Failed to get ColorTestResult class");
    }

    // Get the appropriate enum value (Success or Failure)
    const char* result_name = (previous_result == ColorTestResult::Success) ? "Success" : "Failure";
    PyObject* pPreviousResult = PyObject_GetAttrString(pColorTestResultClass, result_name);
    Py_DECREF(pColorTestResultClass);

    if (!pPreviousResult) {
        PyErr_Print();
        throw std::runtime_error("TestGenerator: Failed to get ColorTestResult enum value");
    }

    // Convert C++ ColorSpaceType enum to Python enum
    PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
    if (!pOutputSpace) {
        Py_DECREF(pPreviousResult);
        throw std::runtime_error("TestGenerator: Failed to convert output_space to Python enum");
    }

    // Call Python GetTest method using PyObject_Call with tuple and kwargs
    // This allows us to use keyword arguments for optional parameters
    PyObject* pMethod = PyObject_GetAttrString(pInstance, "GetTest");
    if (!pMethod) {
        Py_DECREF(pPreviousResult);
        Py_DECREF(pOutputSpace);
        PyErr_Print();
        throw std::runtime_error("TestGenerator: Failed to get GetTest method");
    }

    // Build positional arguments tuple: (previous_result, filename, hidden_symbol)
    PyObject* pArgs = PyTuple_New(3);
    PyTuple_SetItem(pArgs, 0, pPreviousResult); // Steals reference
    PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(filename.c_str()));
    PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(hidden_symbol.c_str()));

    // Build keyword arguments dict: {output_space, lum_noise, s_cone_noise}
    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "output_space", pOutputSpace); // Steals reference
    PyDict_SetItemString(pKwargs, "lum_noise", PyFloat_FromDouble(lum_noise));
    PyDict_SetItemString(pKwargs, "s_cone_noise", PyFloat_FromDouble(s_cone_noise));

    PyObject* pResult = PyObject_Call(pMethod, pArgs, pKwargs);

    Py_DECREF(pMethod);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    // Note: pPreviousResult and pOutputSpace references were stolen, don't DECREF them

    if (!pResult) {
        PyErr_Print();
        throw std::runtime_error("TestGenerator: Failed to call GetTest()");
    }

    // Check if Python returned None (test complete)
    if (pResult == Py_None) {
        Py_DECREF(pResult);
        return std::nullopt;
    }

    // Parse the returned dict to TrialData
    TrialData trial_data = ParseDictToTrialData(pResult);
    Py_DECREF(pResult);

    return trial_data;
}

std::string TestGenerator::GetStringFromDict(
    PyObject* dict,
    const char* key,
    const std::string& default_val
)
{
    PyObject* pValue = PyDict_GetItemString(dict, key);
    if (!pValue) {
        return default_val;
    }

    if (PyUnicode_Check(pValue)) {
        const char* str = PyUnicode_AsUTF8(pValue);
        return str ? std::string(str) : default_val;
    }

    return default_val;
}

int TestGenerator::GetIntFromDict(PyObject* dict, const char* key, int default_val)
{
    PyObject* pValue = PyDict_GetItemString(dict, key);
    if (!pValue) {
        return default_val;
    }

    if (PyLong_Check(pValue)) {
        return PyLong_AsLong(pValue);
    }

    return default_val;
}

double TestGenerator::GetDoubleFromDict(PyObject* dict, const char* key, double default_val)
{
    PyObject* pValue = PyDict_GetItemString(dict, key);
    if (!pValue) {
        return default_val;
    }

    if (PyFloat_Check(pValue)) {
        return PyFloat_AsDouble(pValue);
    } else if (PyLong_Check(pValue)) {
        return static_cast<double>(PyLong_AsLong(pValue));
    }

    return default_val;
}

TrialData TestGenerator::ParseDictToTrialData(PyObject* dict)
{
    if (!PyDict_Check(dict)) {
        throw std::runtime_error("TestGenerator: Expected dict from Python, got something else");
    }

    // Get trial_type to determine which variant to create
    std::string trial_type = GetStringFromDict(dict, "trial_type", "pseudo_isochromatic");

    if (trial_type == "pseudo_isochromatic") {
        PseudoIsochromaticTrial trial;
        trial.genotype = GetStringFromDict(dict, "genotype", "");
        trial.metameric_axis = GetIntFromDict(dict, "metameric_axis", -1);
        trial.rgb_path = GetStringFromDict(dict, "rgb_path", "");
        trial.ocv_path = GetStringFromDict(dict, "ocv_path", "");
        trial.hidden_symbol = GetStringFromDict(dict, "hidden_symbol", "");
        trial.intensity = GetDoubleFromDict(dict, "intensity", 1.0);

        // Extract metadata if present
        PyObject* pMetadata = PyDict_GetItemString(dict, "metadata");
        if (pMetadata && PyDict_Check(pMetadata)) {
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(pMetadata, &pos, &key, &value)) {
                if (PyUnicode_Check(key)) {
                    const char* key_str = PyUnicode_AsUTF8(key);
                    std::string val_str;

                    if (PyUnicode_Check(value)) {
                        val_str = PyUnicode_AsUTF8(value);
                    } else if (PyFloat_Check(value)) {
                        val_str = std::to_string(PyFloat_AsDouble(value));
                    } else if (PyLong_Check(value)) {
                        val_str = std::to_string(PyLong_AsLong(value));
                    } else {
                        // For other types (like lists), convert to string representation
                        PyObject* str_obj = PyObject_Str(value);
                        if (str_obj) {
                            val_str = PyUnicode_AsUTF8(str_obj);
                            Py_DECREF(str_obj);
                        }
                    }

                    if (key_str) {
                        trial.metadata[key_str] = val_str;
                    }
                }
            }
        }

        return trial;
    } else if (trial_type == "circle_grid") {
        CircleGridTrial trial;
        trial.genotype = GetStringFromDict(dict, "genotype", "");
        trial.metameric_axis = GetIntFromDict(dict, "metameric_axis", -1);

        // Extract image_paths (list of strings)
        PyObject* pImagePaths = PyDict_GetItemString(dict, "image_paths");
        if (pImagePaths && PyList_Check(pImagePaths)) {
            Py_ssize_t size = PyList_Size(pImagePaths);
            for (Py_ssize_t i = 0; i < size; i++) {
                PyObject* pPath = PyList_GetItem(pImagePaths, i);
                if (PyUnicode_Check(pPath)) {
                    trial.image_paths.push_back(PyUnicode_AsUTF8(pPath));
                }
            }
        }

        // Extract scramble_indices (list of ints)
        PyObject* pScrambleIndices = PyDict_GetItemString(dict, "scramble_indices");
        if (pScrambleIndices && PyList_Check(pScrambleIndices)) {
            Py_ssize_t size = PyList_Size(pScrambleIndices);
            for (Py_ssize_t i = 0; i < size; i++) {
                PyObject* pIndex = PyList_GetItem(pScrambleIndices, i);
                if (PyLong_Check(pIndex)) {
                    trial.scramble_indices.push_back(PyLong_AsLong(pIndex));
                }
            }
        }

        // Extract metadata if present
        PyObject* pMetadata = PyDict_GetItemString(dict, "metadata");
        if (pMetadata && PyDict_Check(pMetadata)) {
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(pMetadata, &pos, &key, &value)) {
                if (PyUnicode_Check(key) && PyUnicode_Check(value)) {
                    trial.metadata[PyUnicode_AsUTF8(key)] = PyUnicode_AsUTF8(value);
                }
            }
        }

        return trial;
    } else {
        throw std::runtime_error("TestGenerator: Unknown trial_type: " + trial_type);
    }
}

std::vector<std::string> TestGenerator::GetGenotypes()
{
    std::vector<std::string> genotypes;
    if (!pInstance) {
        return genotypes;
    }

    PyObject* pMethod = PyObject_GetAttrString(pInstance, "GetGenotypes");
    if (!pMethod || !PyCallable_Check(pMethod)) {
        Py_XDECREF(pMethod);
        return genotypes; // Method doesn't exist, return empty
    }

    PyObject* pResult = PyObject_CallObject(pMethod, nullptr);
    Py_DECREF(pMethod);

    if (!pResult) {
        PyErr_Print();
        return genotypes;
    }

    // Expect a list of tuples (genotypes are tuples of floats)
    if (PyList_Check(pResult)) {
        Py_ssize_t n = PyList_Size(pResult);
        genotypes.reserve(n);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* genotypeTuple = PyList_GetItem(pResult, i);
            if (PyTuple_Check(genotypeTuple)) {
                // Convert tuple to string representation "(peak1, peak2, ...)"
                PyObject* strRepr = PyObject_Repr(genotypeTuple);
                if (strRepr) {
                    const char* strCStr = PyUnicode_AsUTF8(strRepr);
                    if (strCStr) {
                        genotypes.push_back(std::string(strCStr));
                    }
                    Py_DECREF(strRepr);
                }
            }
        }
    }
    Py_DECREF(pResult);
    return genotypes;
}

int TestGenerator::GetTotalTrials()
{
    if (!pInstance) {
        return -1;
    }

    // Try to get color_generator from TestGenerator instance
    PyObject* pColorGenerator = PyObject_GetAttrString(pInstance, "color_generator");
    if (!pColorGenerator) {
        PyErr_Clear();
        return -1;
    }

    // Try to call get_num_samples() on color generator
    PyObject* pMethod = PyObject_GetAttrString(pColorGenerator, "get_num_samples");
    Py_DECREF(pColorGenerator);

    if (!pMethod || !PyCallable_Check(pMethod)) {
        Py_XDECREF(pMethod);
        return -1;
    }

    PyObject* pResult = PyObject_CallObject(pMethod, nullptr);
    Py_DECREF(pMethod);

    if (!pResult) {
        PyErr_Clear();
        return -1;
    }

    int totalTrials = -1;
    if (PyLong_Check(pResult)) {
        totalTrials = PyLong_AsLong(pResult);
    }
    Py_DECREF(pResult);
    return totalTrials;
}

std::map<int, std::map<std::string, std::string>> TestGenerator::GetThresholds()
{
    std::map<int, std::map<std::string, std::string>> thresholds;
    if (!pInstance) {
        return thresholds;
    }

    // Try to get color_generator from TestGenerator instance
    PyObject* pColorGenerator = PyObject_GetAttrString(pInstance, "color_generator");
    if (!pColorGenerator) {
        PyErr_Clear();
        return thresholds;
    }

    // Try to call get_thresholds() on color generator
    PyObject* pMethod = PyObject_GetAttrString(pColorGenerator, "get_thresholds");
    Py_DECREF(pColorGenerator);

    if (!pMethod || !PyCallable_Check(pMethod)) {
        Py_XDECREF(pMethod);
        return thresholds;
    }

    PyObject* pResult = PyObject_CallObject(pMethod, nullptr);
    Py_DECREF(pMethod);

    if (!pResult) {
        PyErr_Clear();
        return thresholds;
    }

    // Parse the returned dict
    if (PyDict_Check(pResult)) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(pResult, &pos, &key, &value)) {
            if (PyLong_Check(key) && PyDict_Check(value)) {
                int directionIdx = PyLong_AsLong(key);
                std::map<std::string, std::string> directionData;

                // Parse nested dict
                PyObject *innerKey, *innerValue;
                Py_ssize_t innerPos = 0;
                while (PyDict_Next(value, &innerPos, &innerKey, &innerValue)) {
                    if (PyUnicode_Check(innerKey)) {
                        const char* keyStr = PyUnicode_AsUTF8(innerKey);
                        std::string valStr;

                        // Convert value to string
                        if (PyUnicode_Check(innerValue)) {
                            valStr = PyUnicode_AsUTF8(innerValue);
                        } else if (PyFloat_Check(innerValue)) {
                            valStr = std::to_string(PyFloat_AsDouble(innerValue));
                        } else if (PyLong_Check(innerValue)) {
                            valStr = std::to_string(PyLong_AsLong(innerValue));
                        } else if (PyList_Check(innerValue) || PyTuple_Check(innerValue)) {
                            // Convert list/tuple to string representation
                            PyObject* strObj = PyObject_Str(innerValue);
                            if (strObj) {
                                valStr = PyUnicode_AsUTF8(strObj);
                                Py_DECREF(strObj);
                            }
                        } else {
                            PyObject* strObj = PyObject_Str(innerValue);
                            if (strObj) {
                                valStr = PyUnicode_AsUTF8(strObj);
                                Py_DECREF(strObj);
                            }
                        }

                        if (keyStr) {
                            directionData[keyStr] = valStr;
                        }
                    }
                }
                thresholds[directionIdx] = directionData;
            }
        }
    }
    Py_DECREF(pResult);
    return thresholds;
}

bool TestGenerator::ExportThresholds(const std::string& filename)
{
    if (!pInstance) {
        return false;
    }

    // Try to get color_generator from TestGenerator instance
    PyObject* pColorGenerator = PyObject_GetAttrString(pInstance, "color_generator");
    if (!pColorGenerator) {
        PyErr_Clear();
        return false;
    }

    // Try to call export_thresholds() on color generator
    PyObject* pMethod = PyObject_GetAttrString(pColorGenerator, "export_thresholds");
    Py_DECREF(pColorGenerator);

    if (!pMethod || !PyCallable_Check(pMethod)) {
        Py_XDECREF(pMethod);
        return false;
    }

    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(filename.c_str()));

    PyObject* pResult = PyObject_CallObject(pMethod, pArgs);
    Py_DECREF(pMethod);
    Py_DECREF(pArgs);

    bool success = (pResult != nullptr);
    if (pResult) {
        Py_DECREF(pResult);
    } else {
        PyErr_Clear();
    }
    return success;
}

} // namespace TetriumColor
