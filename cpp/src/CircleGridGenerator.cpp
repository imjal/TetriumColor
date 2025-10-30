#include "TetriumColor/TetriumColor.h"
#include <cstdlib>
#include <ctime>

namespace TetriumColor
{

static PyObject* ColorSpaceTypeToPython(ColorSpaceType space_type)
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
        type_name = "SRGB";
        break;
    case ColorSpaceType::OCV:
        type_name = "DISP_6P";
        break;
    default:
        type_name = "DISP_6P";
        break;
    }
    PyObject* pValue = PyObject_GetAttrString(pColorSpaceType, type_name);
    Py_DECREF(pColorSpaceType);
    return pValue;
}

CircleGridGenerator::CircleGridGenerator(
    const std::string& primary_path,
    int num_samples,
    float scramble_prob
)
{
    PyObject* pName = PyUnicode_DecodeFSDefault("TetriumColor.TetraColorPicker");
    pModule = reinterpret_cast<PyObject*>(PyImport_Import(pName));
    Py_DECREF(pName);
    if (pModule != nullptr) {
        pClass = reinterpret_cast<PyObject*>(
            PyObject_GetAttrString(reinterpret_cast<PyObject*>(pModule), "CircleGridGenerator")
        );
        if (pClass && PyCallable_Check(reinterpret_cast<PyObject*>(pClass)) == 1) {
            PyObject* pArgs = PyTuple_Pack(
                3,
                PyUnicode_FromString(primary_path.c_str()),
                PyLong_FromLong(num_samples),
                PyFloat_FromDouble(scramble_prob)
            );
            pInstance = reinterpret_cast<PyObject*>(
                PyObject_CallObject(reinterpret_cast<PyObject*>(pClass), pArgs)
            );
            Py_DECREF(pArgs);
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
}

CircleGridGenerator::~CircleGridGenerator()
{
    Py_XDECREF(reinterpret_cast<PyObject*>(pInstance));
    Py_XDECREF(reinterpret_cast<PyObject*>(pClass));
    Py_XDECREF(reinterpret_cast<PyObject*>(pModule));
}

std::vector<std::pair<int, int>> CircleGridGenerator::GetImages(
    int metameric_axis,
    float luminance,
    float saturation,
    const std::vector<std::string>& filenames,
    ColorSpaceType output_space,
    int seed
)
{
    std::vector<std::pair<int, int>> idxs;
    if (!pInstance)
        return idxs;

    PyObject* pList = PyList_New(filenames.size());
    for (size_t i = 0; i < filenames.size(); ++i) {
        PyList_SetItem(pList, i, PyUnicode_FromString(filenames[i].c_str()));
    }

    PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
    if (!pOutputSpace)
        return idxs;

    // If seed is -1, generate a random seed
    if (seed == -1) {
        seed = static_cast<int>(time(nullptr)) + rand();
    }

    PyObject* pValue = PyObject_CallMethod(
        reinterpret_cast<PyObject*>(pInstance),
        "GetImages",
        "iffOOi",
        metameric_axis,
        luminance,
        saturation,
        pList,
        pOutputSpace,
        seed
    );
    Py_DECREF(pList);
    Py_DECREF(pOutputSpace);

    if (!pValue) {
        PyErr_Print();
        return idxs;
    }

    // Expect a list of tuples
    if (PyList_Check(pValue)) {
        Py_ssize_t n = PyList_Size(pValue);
        idxs.reserve(n);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* tup = PyList_GetItem(pValue, i);
            if (PyTuple_Check(tup) && PyTuple_Size(tup) == 2) {
                int first = static_cast<int>(PyLong_AsLong(PyTuple_GetItem(tup, 0)));
                int second = static_cast<int>(PyLong_AsLong(PyTuple_GetItem(tup, 1)));
                idxs.emplace_back(first, second);
            }
        }
    }
    Py_DECREF(pValue);
    return idxs;
}

} // namespace TetriumColor
