#include "TetriumColor/ColorGenerator.h"
#include <cstdio>
#include <vector>

namespace TetriumColor
{
ColorGenerator::ColorGenerator(
    const std::string& sex,
    float percentage_screened,
    float peak_to_test,
    const std::vector<int>& dimensions,
    const std::string& display_primaries_path
)
{
    // Import required Python modules
    PyObject* pMeasurementModule = PyImport_ImportModule("TetriumColor.Measurement");
    if (!pMeasurementModule) {
        PyErr_Print();
        exit(-1);
    }

    // Load primaries from CSV
    PyObject* pLoadPrimariesFunc
        = PyObject_GetAttrString(pMeasurementModule, "load_primaries_from_csv");
    printf("pLoadPrimariesFunc: %p\n", pLoadPrimariesFunc);
    PyObject* pPrimaries
        = PyObject_CallFunction(pLoadPrimariesFunc, "s", display_primaries_path.c_str());
    printf("pPrimaries: %p\n", pPrimaries);
    Py_DECREF(pLoadPrimariesFunc);
    Py_DECREF(pMeasurementModule);

    if (!pPrimaries) {
        PyErr_Print();
        printf("Failed to load primaries from %s\n", display_primaries_path.c_str());
        exit(-1);
    }

    // Import TetraColorPicker module
    PyObject* pName = PyUnicode_DecodeFSDefault("TetriumColor.TetraColorPicker");
    pModule = reinterpret_cast<PyObject*>(PyImport_Import(pName));
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the GeneticCDFTestColorGenerator class
        pClass = reinterpret_cast<PyObject*>(PyObject_GetAttrString(
            reinterpret_cast<PyObject*>(pModule), "GeneticCDFTestColorGenerator"
        ));

        if (pClass && PyCallable_Check(reinterpret_cast<PyObject*>(pClass)) == 1) {
            // Convert dimensions vector to Python list
            PyObject* py_dimensions = PyList_New(dimensions.size());
            for (size_t i = 0; i < dimensions.size(); ++i) {
                PyList_SetItem(py_dimensions, i, PyLong_FromLong(dimensions[i]));
            }

            // Create instance with keyword arguments
            PyObject* pArgs = PyTuple_New(0);
            PyObject* pKwargs = PyDict_New();
            PyDict_SetItemString(pKwargs, "sex", PyUnicode_FromString(sex.c_str()));
            PyDict_SetItemString(
                pKwargs, "percentage_screened", PyFloat_FromDouble(percentage_screened)
            );
            PyDict_SetItemString(pKwargs, "peak_to_test", PyFloat_FromDouble(peak_to_test));
            PyDict_SetItemString(pKwargs, "dimensions", py_dimensions);
            PyDict_SetItemString(pKwargs, "display_primaries", pPrimaries);

            pInstance = reinterpret_cast<PyObject*>(
                PyObject_Call(reinterpret_cast<PyObject*>(pClass), pArgs, pKwargs)
            );

            Py_DECREF(pArgs);
            Py_DECREF(pKwargs);
            Py_DECREF(py_dimensions);

            if (!pInstance) {
                PyErr_Print();
                exit(-1);
            }
        } else {
            PyErr_Print();
            exit(-1);
        }
    } else {
        PyErr_Print();
        exit(-1);
    }

    Py_DECREF(pPrimaries);
}

ColorGenerator::~ColorGenerator()
{
    Py_XDECREF(pInstance);
    Py_XDECREF(pClass);
    Py_XDECREF(pModule);
}

int ColorGenerator::GetNumSamples()
{
    if (pInstance != nullptr) {
        PyObject* pValue = PyObject_CallMethod(
            reinterpret_cast<PyObject*>(pInstance), "get_num_samples", nullptr
        );
        if (pValue != nullptr) {
            int num_samples = PyLong_AsLong(pValue);
            Py_DECREF(pValue);
            return num_samples;
        } else {
            PyErr_Print();
            return 0;
        }
    }
    return 0;
}

} // namespace TetriumColor
