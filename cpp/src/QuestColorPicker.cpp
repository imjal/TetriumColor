#include "TetriumColor/QuestColorPicker.h"
#include <Python.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace TetriumColor
{

QuestColorPicker::QuestColorPicker(
    const std::string& mode,
    int num_genotypes,
    int trials_per_direction,
    const std::string& sex,
    float background_luminance,
    int seed,
    const std::string& display_primaries_path,
    const std::vector<int>& metameric_axes
)
    : pModule(nullptr), pClass(nullptr), pInstance(nullptr)
{
    // Load display primaries if provided
    PyObject* pPrimaries = nullptr;
    if (!display_primaries_path.empty()) {
        PyObject* pMeasurementModule = PyImport_ImportModule("TetriumColor.Measurement");
        if (!pMeasurementModule) {
            PyErr_Print();
            throw std::runtime_error("Failed to import TetriumColor.Measurement module");
        }

        PyObject* pLoadFunc = PyObject_GetAttrString(pMeasurementModule, "load_primaries_from_csv");
        if (pLoadFunc && PyCallable_Check(pLoadFunc)) {
            PyObject* pPrimariesArgs = PyTuple_New(1);
            PyTuple_SetItem(
                pPrimariesArgs, 0, PyUnicode_FromString(display_primaries_path.c_str())
            );
            pPrimaries = PyObject_CallObject(pLoadFunc, pPrimariesArgs);
            Py_DECREF(pPrimariesArgs);
            Py_DECREF(pLoadFunc);
        }
        Py_DECREF(pMeasurementModule);

        if (!pPrimaries) {
            PyErr_Print();
            throw std::runtime_error("Failed to load primaries from path");
        }
    }

    // Create Observer (tetrachromat)
    PyObject* pObserverModule = PyImport_ImportModule("TetriumColor.Observer");
    if (!pObserverModule) {
        Py_XDECREF(pPrimaries);
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.Observer module");
    }

    PyObject* pObserverClass = PyObject_GetAttrString(pObserverModule, "Observer");
    PyObject* pTetrachromatMethod = PyObject_GetAttrString(pObserverClass, "tetrachromat");
    PyObject* pObserver = PyObject_CallObject(pTetrachromatMethod, nullptr);

    Py_DECREF(pTetrachromatMethod);
    Py_DECREF(pObserverClass);
    Py_DECREF(pObserverModule);

    if (!pObserver) {
        Py_XDECREF(pPrimaries);
        PyErr_Print();
        throw std::runtime_error("Failed to create Observer");
    }

    // Create ColorSpace instance with observer
    PyObject* pColorSpaceModule = PyImport_ImportModule("TetriumColor.ColorSpace");
    if (!pColorSpaceModule) {
        Py_DECREF(pObserver);
        Py_XDECREF(pPrimaries);
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.ColorSpace module");
    }

    PyObject* pColorSpaceClass = PyObject_GetAttrString(pColorSpaceModule, "ColorSpace");
    Py_DECREF(pColorSpaceModule);

    if (!pColorSpaceClass || !PyCallable_Check(pColorSpaceClass)) {
        Py_XDECREF(pColorSpaceClass);
        Py_DECREF(pObserver);
        Py_XDECREF(pPrimaries);
        throw std::runtime_error("Cannot find ColorSpace class");
    }

    // ColorSpace(observer, display_primaries=...)
    PyObject* pCSArgs = PyTuple_New(1);
    PyTuple_SetItem(pCSArgs, 0, pObserver); // steals reference to pObserver

    PyObject* pCSKwargs = PyDict_New();
    if (pPrimaries) {
        PyDict_SetItemString(pCSKwargs, "display_primaries", pPrimaries);
        Py_DECREF(pPrimaries);
    }

    PyObject* pColorSpaceInstance = PyObject_Call(pColorSpaceClass, pCSArgs, pCSKwargs);
    Py_DECREF(pCSArgs);
    Py_DECREF(pCSKwargs);
    Py_DECREF(pColorSpaceClass);

    if (!pColorSpaceInstance) {
        PyErr_Print();
        throw std::runtime_error("Failed to create ColorSpace instance");
    }

    // Import the TetriumColor.TetraColorPicker module
    PyObject* pModuleName = PyUnicode_FromString("TetriumColor.TetraColorPicker");
    pModule = PyImport_Import(pModuleName);
    Py_DECREF(pModuleName);

    if (pModule == nullptr) {
        Py_DECREF(pColorSpaceInstance);
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.TetraColorPicker module");
    }

    // Get the QuestColorGenerator class
    pClass = PyObject_GetAttrString((PyObject*)pModule, "QuestColorGenerator");
    if (pClass == nullptr || !PyCallable_Check((PyObject*)pClass)) {
        Py_XDECREF((PyObject*)pClass);
        Py_DECREF((PyObject*)pModule);
        Py_DECREF(pColorSpaceInstance);
        throw std::runtime_error("Cannot find QuestColorGenerator class");
    }

    // Create arguments tuple
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pColorSpaceInstance); // steals reference

    PyObject* pKwargs = PyDict_New();

    // Add keyword arguments
    PyDict_SetItemString(pKwargs, "mode", PyUnicode_FromString(mode.c_str()));
    PyDict_SetItemString(pKwargs, "num_genotypes", PyLong_FromLong(num_genotypes));
    PyDict_SetItemString(pKwargs, "trials_per_direction", PyLong_FromLong(trials_per_direction));
    PyDict_SetItemString(pKwargs, "sex", PyUnicode_FromString(sex.c_str()));
    PyDict_SetItemString(pKwargs, "background_luminance", PyFloat_FromDouble(background_luminance));

    // Add metameric_axes if specified
    if (!metameric_axes.empty()) {
        PyObject* pAxesList = PyList_New(metameric_axes.size());
        for (size_t i = 0; i < metameric_axes.size(); i++) {
            PyList_SetItem(pAxesList, i, PyLong_FromLong(metameric_axes[i]));
        }
        PyDict_SetItemString(pKwargs, "metameric_axes", pAxesList);
        Py_DECREF(pAxesList);
    }

    // Create the instance
    pInstance = PyObject_Call((PyObject*)pClass, pArgs, pKwargs);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);

    if (pInstance == nullptr) {
        PyErr_Print();
        Py_DECREF((PyObject*)pClass);
        Py_DECREF((PyObject*)pModule);
        throw std::runtime_error("Failed to create QuestColorGenerator instance");
    }
}

QuestColorPicker::~QuestColorPicker()
{
    Py_XDECREF((PyObject*)pInstance);
    Py_XDECREF((PyObject*)pClass);
    Py_XDECREF((PyObject*)pModule);
}

std::map<int, std::pair<std::string, int>> QuestColorPicker::GetDirectionsMetadata() const
{
    std::map<int, std::pair<std::string, int>> result;

    PyObject* pDirectionMetadata
        = PyObject_GetAttrString((PyObject*)pInstance, "direction_metadata");
    if (pDirectionMetadata == nullptr || !PyList_Check(pDirectionMetadata)) {
        Py_XDECREF(pDirectionMetadata);
        return result;
    }

    Py_ssize_t size = PyList_Size(pDirectionMetadata);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* pMetadata = PyList_GetItem(pDirectionMetadata, i); // borrowed reference
        if (!PyDict_Check(pMetadata)) {
            continue;
        }

        // Get genotype (if exists)
        PyObject* pGenotype = PyDict_GetItemString(pMetadata, "genotype");            // borrowed
        PyObject* pMetamericAxis = PyDict_GetItemString(pMetadata, "metameric_axis"); // borrowed

        if (pGenotype != nullptr && pGenotype != Py_None && pMetamericAxis != nullptr
            && pMetamericAxis != Py_None) {
            // Convert genotype tuple to string
            std::string genotype_str = "(";
            if (PyTuple_Check(pGenotype)) {
                Py_ssize_t genotype_size = PyTuple_Size(pGenotype);
                for (Py_ssize_t j = 0; j < genotype_size; j++) {
                    PyObject* pPeak = PyTuple_GetItem(pGenotype, j); // borrowed
                    if (PyFloat_Check(pPeak)) {
                        if (j > 0)
                            genotype_str += ",";
                        int peak_int = (int)PyFloat_AsDouble(pPeak);
                        genotype_str += std::to_string(peak_int);
                    } else if (PyLong_Check(pPeak)) {
                        if (j > 0)
                            genotype_str += ",";
                        genotype_str += std::to_string(PyLong_AsLong(pPeak));
                    }
                }
            }
            genotype_str += ")";

            int metameric_axis = PyLong_AsLong(pMetamericAxis);
            result[i] = std::make_pair(genotype_str, metameric_axis);
        }
    }

    Py_DECREF(pDirectionMetadata);
    return result;
}

size_t QuestColorPicker::GetNumDirections() const
{
    PyObject* pDirections = PyObject_GetAttrString((PyObject*)pInstance, "directions");
    if (pDirections == nullptr || !PyList_Check(pDirections)) {
        Py_XDECREF(pDirections);
        return 0;
    }

    Py_ssize_t size = PyList_Size(pDirections);
    Py_DECREF(pDirections);
    return (size_t)size;
}

QuestColorResult QuestColorPicker::NewColor()
{
    // Call Python NewColor() method
    PyObject* pNewColorMethod = PyObject_GetAttrString((PyObject*)pInstance, "NewColor");
    if (pNewColorMethod == nullptr || !PyCallable_Check(pNewColorMethod)) {
        Py_XDECREF(pNewColorMethod);
        throw std::runtime_error("NewColor method not found");
    }

    PyObject* pResult = PyObject_CallObject(pNewColorMethod, nullptr);
    Py_DECREF(pNewColorMethod);

    if (pResult == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to call NewColor");
    }

    // Result is (background_cone, test_cone, color_space, distance)
    // We just need to extract the current direction and intensity
    Py_DECREF(pResult);

    // Get current direction index
    PyObject* pCurrentDir = PyObject_GetAttrString((PyObject*)pInstance, "current_direction_idx");
    if (pCurrentDir == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to get current_direction_idx");
    }
    current_direction_idx = PyLong_AsLong(pCurrentDir);
    Py_DECREF(pCurrentDir);

    // Get _last_intensity (the intensity that was used)
    PyObject* pLastIntensity = PyObject_GetAttrString((PyObject*)pInstance, "_last_intensity");
    if (pLastIntensity == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to get _last_intensity");
    }
    double log_intensity = PyFloat_AsDouble(pLastIntensity);
    Py_DECREF(pLastIntensity);

    // Convert from log10 to actual proportion
    double intensity = pow(10.0, log_intensity);
    intensity = std::max(0.0, std::min(1.0, intensity));

    // Get metadata for this direction
    auto metadata = GetDirectionsMetadata();
    auto it = metadata.find(current_direction_idx);
    if (it == metadata.end()) {
        // Debug: print available keys
        std::string available_keys = "";
        for (const auto& [key, value] : metadata) {
            available_keys += std::to_string(key) + " ";
        }
        throw std::runtime_error(
            "NewColor: Direction metadata not found for direction_idx "
            + std::to_string(current_direction_idx) + ". Available direction indices: ["
            + available_keys + "]"
        );
    }

    QuestColorResult result;
    result.direction_idx = current_direction_idx;
    result.genotype = it->second.first;
    result.metameric_axis = it->second.second;
    result.intensity = intensity;
    result.is_done = false;

    return result;
}

QuestColorResult QuestColorPicker::GetColor(bool correct)
{
    // Import ColorTestResult enum
    PyObject* pUtilsModule = PyImport_ImportModule("TetriumColor.Utils.CustomTypes");
    if (!pUtilsModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.Utils.CustomTypes");
    }

    PyObject* pColorTestResult = PyObject_GetAttrString(pUtilsModule, "ColorTestResult");
    Py_DECREF(pUtilsModule);
    if (!pColorTestResult) {
        PyErr_Print();
        throw std::runtime_error("Failed to get ColorTestResult enum");
    }

    // Get Success or Failure enum value
    const char* result_name = correct ? "Success" : "Failure";
    PyObject* pResultValue = PyObject_GetAttrString(pColorTestResult, result_name);
    Py_DECREF(pColorTestResult);
    if (!pResultValue) {
        PyErr_Print();
        throw std::runtime_error("Failed to get ColorTestResult value");
    }

    // Call Python GetColor(previous_result) method
    PyObject* pGetColorMethod = PyObject_GetAttrString((PyObject*)pInstance, "GetColor");
    if (pGetColorMethod == nullptr || !PyCallable_Check(pGetColorMethod)) {
        Py_XDECREF(pGetColorMethod);
        Py_DECREF(pResultValue);
        throw std::runtime_error("GetColor method not found");
    }

    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pResultValue); // steals reference

    PyObject* pResult = PyObject_CallObject(pGetColorMethod, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pGetColorMethod);

    if (pResult == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to call GetColor");
    }

    QuestColorResult result;

    // Check if result is None (all trials complete)
    if (pResult == Py_None) {
        Py_DECREF(pResult);
        result.is_done = true;
        return result;
    }

    // Result is (background_cone, test_cone, color_space, distance)
    Py_DECREF(pResult);

    // Get current direction index
    PyObject* pCurrentDir = PyObject_GetAttrString((PyObject*)pInstance, "current_direction_idx");
    if (pCurrentDir == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to get current_direction_idx");
    }
    current_direction_idx = PyLong_AsLong(pCurrentDir);
    Py_DECREF(pCurrentDir);

    // Get _last_intensity (the intensity that was used)
    PyObject* pLastIntensity = PyObject_GetAttrString((PyObject*)pInstance, "_last_intensity");
    if (pLastIntensity == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to get _last_intensity");
    }
    double log_intensity = PyFloat_AsDouble(pLastIntensity);
    Py_DECREF(pLastIntensity);

    // Convert from log10 to actual proportion
    double intensity = pow(10.0, log_intensity);
    intensity = std::max(0.0, std::min(1.0, intensity));

    // Get metadata for this direction
    auto metadata = GetDirectionsMetadata();
    auto it = metadata.find(current_direction_idx);
    if (it == metadata.end()) {
        throw std::runtime_error("Direction metadata not found");
    }

    result.direction_idx = current_direction_idx;
    result.genotype = it->second.first;
    result.metameric_axis = it->second.second;
    result.intensity = intensity;
    result.is_done = false;

    return result;
}

void QuestColorPicker::ExportThresholds(const std::string& filename)
{
    PyObject* pMethod = PyObject_GetAttrString((PyObject*)pInstance, "export_thresholds");
    if (pMethod == nullptr || !PyCallable_Check(pMethod)) {
        Py_XDECREF(pMethod);
        throw std::runtime_error("export_thresholds method not found");
    }

    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(filename.c_str()));

    PyObject* pResult = PyObject_CallObject(pMethod, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pMethod);

    if (pResult == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to export thresholds");
    }

    Py_DECREF(pResult);
}

} // namespace TetriumColor
