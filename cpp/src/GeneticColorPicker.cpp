#include "TetriumColor/GeneticColorPicker.h"
#include <Python.h>
#include <iostream>
#include <stdexcept>

namespace TetriumColor
{

GeneticColorPicker::GeneticColorPicker(
    const std::string& sex,
    float percentage_screened,
    float peak_to_test,
    float luminance,
    float saturation,
    const std::vector<int>& dimensions,
    int seed,
    const std::string& display_primaries_path
)
    : pModule(nullptr), pClass(nullptr), pInstance(nullptr)
{
    // Import the TetriumColor.TetraColorPicker module
    PyObject* pModuleName = PyUnicode_FromString("TetriumColor.TetraColorPicker");
    pModule = PyImport_Import(pModuleName);
    Py_DECREF(pModuleName);

    if (pModule == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.TetraColorPicker module");
    }

    // Get the GeneticColorPicker class
    pClass = PyObject_GetAttrString((PyObject*)pModule, "GeneticColorPicker");
    if (pClass == nullptr || !PyCallable_Check((PyObject*)pClass)) {
        Py_XDECREF((PyObject*)pClass);
        Py_DECREF((PyObject*)pModule);
        throw std::runtime_error("Cannot find GeneticColorPicker class");
    }

    // Create arguments tuple
    PyObject* pArgs = PyTuple_New(0);
    PyObject* pKwargs = PyDict_New();

    // Add keyword arguments
    PyDict_SetItemString(pKwargs, "sex", PyUnicode_FromString(sex.c_str()));
    PyDict_SetItemString(pKwargs, "percentage_screened", PyFloat_FromDouble(percentage_screened));
    PyDict_SetItemString(pKwargs, "peak_to_test", PyFloat_FromDouble(peak_to_test));
    PyDict_SetItemString(pKwargs, "luminance", PyFloat_FromDouble(luminance));
    PyDict_SetItemString(pKwargs, "saturation", PyFloat_FromDouble(saturation));
    PyDict_SetItemString(pKwargs, "seed", PyLong_FromLong(seed));

    // Add dimensions list
    PyObject* pDimensions = PyList_New(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        PyList_SetItem(pDimensions, i, PyLong_FromLong(dimensions[i]));
    }
    PyDict_SetItemString(pKwargs, "dimensions", pDimensions);

    // Add display_primaries_path if provided
    if (!display_primaries_path.empty()) {
        // Load primaries using load_primaries_from_csv
        PyObject* pMeasurementModule = PyImport_ImportModule("TetriumColor.Measurement");
        if (pMeasurementModule) {
            PyObject* pLoadFunc
                = PyObject_GetAttrString(pMeasurementModule, "load_primaries_from_csv");
            if (pLoadFunc && PyCallable_Check(pLoadFunc)) {
                PyObject* pPrimariesArgs = PyTuple_New(1);
                PyTuple_SetItem(
                    pPrimariesArgs, 0, PyUnicode_FromString(display_primaries_path.c_str())
                );
                PyObject* pPrimaries = PyObject_CallObject(pLoadFunc, pPrimariesArgs);
                if (pPrimaries) {
                    PyDict_SetItemString(pKwargs, "display_primaries", pPrimaries);
                    Py_DECREF(pPrimaries);
                }
                Py_DECREF(pPrimariesArgs);
                Py_DECREF(pLoadFunc);
            }
            Py_DECREF(pMeasurementModule);
        }
    }

    // Create the instance
    pInstance = PyObject_Call((PyObject*)pClass, pArgs, pKwargs);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);

    if (pInstance == nullptr) {
        PyErr_Print();
        Py_DECREF((PyObject*)pClass);
        Py_DECREF((PyObject*)pModule);
        throw std::runtime_error("Failed to create GeneticColorPicker instance");
    }
}

GeneticColorPicker::~GeneticColorPicker()
{
    Py_XDECREF((PyObject*)pInstance);
    Py_XDECREF((PyObject*)pClass);
    Py_XDECREF((PyObject*)pModule);
}

std::vector<std::string> GeneticColorPicker::GetGenotypes() const
{
    std::vector<std::string> genotypes;

    PyObject* pGetGenotypes = PyObject_GetAttrString((PyObject*)pInstance, "GetGenotypes");
    if (pGetGenotypes && PyCallable_Check(pGetGenotypes)) {
        PyObject* pResult = PyObject_CallObject(pGetGenotypes, nullptr);
        if (pResult) {
            if (PyList_Check(pResult)) {
                Py_ssize_t size = PyList_Size(pResult);
                for (Py_ssize_t i = 0; i < size; ++i) {
                    PyObject* pGenotype = PyList_GetItem(pResult, i);
                    PyObject* pStr = PyObject_Str(pGenotype);
                    const char* str = PyUnicode_AsUTF8(pStr);
                    if (str) {
                        genotypes.push_back(std::string(str));
                    }
                    Py_DECREF(pStr);
                }
            }
            Py_DECREF(pResult);
        }
        Py_DECREF(pGetGenotypes);
    }

    return genotypes;
}

size_t GeneticColorPicker::GetNumGenotypes() const { return GetGenotypes().size(); }

} // namespace TetriumColor
