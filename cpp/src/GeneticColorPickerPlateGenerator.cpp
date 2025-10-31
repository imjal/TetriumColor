#include "TetriumColor/GeneticColorPickerPlateGenerator.h"
#include <Python.h>
#include <iostream>
#include <stdexcept>

namespace TetriumColor
{

// Helper function to convert ColorSpaceType enum to Python enum
static PyObject* ColorSpaceTypeToPython(ColorSpaceType space_type)
{
    // Import the ColorSpaceType enum from Python
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

GeneticColorPickerPlateGenerator::GeneticColorPickerPlateGenerator(
    GeneticColorPicker& color_picker,
    int seed
)
    : pModule(nullptr), pClass(nullptr), pInstance(nullptr), colorPicker(color_picker)
{
    // Import the TetriumColor.TetraPlate module
    PyObject* pModuleName = PyUnicode_FromString("TetriumColor.TetraPlate");
    pModule = PyImport_Import(pModuleName);
    Py_DECREF(pModuleName);

    if (pModule == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.TetraPlate module");
    }

    // Get the GeneticColorPickerPlateGenerator class
    pClass = PyObject_GetAttrString((PyObject*)pModule, "GeneticColorPickerPlateGenerator");
    if (pClass == nullptr || !PyCallable_Check((PyObject*)pClass)) {
        Py_XDECREF((PyObject*)pClass);
        Py_DECREF((PyObject*)pModule);
        throw std::runtime_error("Cannot find GeneticColorPickerPlateGenerator class");
    }

    // Get the Python instance of the color picker
    PyObject* pColorPickerInstance = (PyObject*)color_picker.GetPythonInstance();

    // Create arguments tuple with the color picker instance
    PyObject* pArgs = PyTuple_New(1);
    Py_INCREF(pColorPickerInstance); // Need to increment before PyTuple_SetItem steals it
    PyTuple_SetItem(pArgs, 0, pColorPickerInstance);

    // Create the instance
    pInstance = PyObject_CallObject((PyObject*)pClass, pArgs);
    Py_DECREF(pArgs);

    if (pInstance == nullptr) {
        PyErr_Print();
        Py_DECREF((PyObject*)pClass);
        Py_DECREF((PyObject*)pModule);
        throw std::runtime_error("Failed to create GeneticColorPickerPlateGenerator instance");
    }
}

GeneticColorPickerPlateGenerator::~GeneticColorPickerPlateGenerator()
{
    Py_XDECREF((PyObject*)pInstance);
    Py_XDECREF((PyObject*)pClass);
    Py_XDECREF((PyObject*)pModule);
}

void GeneticColorPickerPlateGenerator::GetPlate(
    const std::string& genotype,
    int metameric_axis,
    const std::string& filename,
    int hidden_number,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise
)
{
    PyObject* pGetPlate = PyObject_GetAttrString((PyObject*)pInstance, "GetPlate");
    if (!pGetPlate || !PyCallable_Check(pGetPlate)) {
        Py_XDECREF(pGetPlate);
        throw std::runtime_error("Cannot find GetPlate method");
    }

    // Parse genotype string to tuple
    // Format is like "(558.9, 530.3)"
    PyObject* pGenotype = PyRun_String(genotype.c_str(), Py_eval_input, PyDict_New(), PyDict_New());
    if (!pGenotype) {
        PyErr_Print();
        Py_DECREF(pGetPlate);
        throw std::runtime_error("Failed to parse genotype string");
    }

    PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
    if (!pOutputSpace) {
        printf("Failed to convert ColorSpaceType to Python\n");
        return;
    }

    // Create arguments
    PyObject* pArgs = PyTuple_New(5);
    PyTuple_SetItem(pArgs, 0, pGenotype);
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(metameric_axis));
    PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(filename.c_str()));
    PyTuple_SetItem(pArgs, 3, PyLong_FromLong(hidden_number));
    PyTuple_SetItem(pArgs, 4, pOutputSpace);

    // Create keyword arguments for optional parameters
    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "lum_noise", PyFloat_FromDouble(lum_noise));
    PyDict_SetItemString(pKwargs, "s_cone_noise", PyFloat_FromDouble(s_cone_noise));

    // Call the method
    PyObject* pResult = PyObject_Call(pGetPlate, pArgs, pKwargs);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pGetPlate);
    Py_DECREF(pOutputSpace);

    if (pResult == nullptr) {
        PyErr_Print();
        throw std::runtime_error("GetPlate failed");
    }

    Py_DECREF(pResult);
}

void GeneticColorPickerPlateGenerator::GetPlate(
    const std::string& genotype,
    int metameric_axis,
    const std::string& filename,
    const std::string& hidden_symbol,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise
)
{
    PyObject* pGetPlate = PyObject_GetAttrString((PyObject*)pInstance, "GetPlate");
    if (!pGetPlate || !PyCallable_Check(pGetPlate)) {
        Py_XDECREF(pGetPlate);
        throw std::runtime_error("Cannot find GetPlate method");
    }

    // Parse genotype string to tuple
    PyObject* pGenotype = PyRun_String(genotype.c_str(), Py_eval_input, PyDict_New(), PyDict_New());
    if (!pGenotype) {
        PyErr_Print();
        Py_DECREF(pGetPlate);
        throw std::runtime_error("Failed to parse genotype string");
    }

    PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
    if (!pOutputSpace) {
        printf("Failed to convert ColorSpaceType to Python\n");
        return;
    }

    // Create arguments
    PyObject* pArgs = PyTuple_New(5);
    PyTuple_SetItem(pArgs, 0, pGenotype);
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(metameric_axis));
    PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(filename.c_str()));
    PyTuple_SetItem(pArgs, 3, PyUnicode_FromString(hidden_symbol.c_str()));
    PyTuple_SetItem(pArgs, 4, pOutputSpace);

    // Create keyword arguments for optional parameters
    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "lum_noise", PyFloat_FromDouble(lum_noise));
    PyDict_SetItemString(pKwargs, "s_cone_noise", PyFloat_FromDouble(s_cone_noise));

    // Call the method
    PyObject* pResult = PyObject_Call(pGetPlate, pArgs, pKwargs);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pGetPlate);
    Py_DECREF(pOutputSpace);

    if (pResult == nullptr) {
        PyErr_Print();
        throw std::runtime_error("GetPlate failed");
    }

    Py_DECREF(pResult);
}

} // namespace TetriumColor
