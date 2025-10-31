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
    float scramble_prob,
    const std::string& sex,
    float percentage_screened,
    float peak_to_test,
    float luminance,
    float saturation,
    const std::vector<int>& dimensions,
    int seed,
    const std::string& cst_display_type,
    const std::string& display_primaries_path
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
            // Create dimensions list
            PyObject* pDimensions = PyList_New(dimensions.size());
            for (size_t i = 0; i < dimensions.size(); ++i) {
                PyList_SetItem(pDimensions, i, PyLong_FromLong(dimensions[i]));
            }

            // Load primaries from path if provided
            PyObject* pPrimaries = Py_None;
            Py_INCREF(Py_None);
            if (!display_primaries_path.empty()) {
                PyObject* pMeasurementModule = PyImport_ImportModule("TetriumColor.Measurement");
                if (pMeasurementModule) {
                    PyObject* pLoadFunc
                        = PyObject_GetAttrString(pMeasurementModule, "load_primaries_from_csv");
                    if (pLoadFunc && PyCallable_Check(pLoadFunc)) {
                        pPrimaries
                            = PyObject_CallFunction(pLoadFunc, "s", display_primaries_path.c_str());
                        Py_DECREF(pLoadFunc);
                    }
                    Py_DECREF(pMeasurementModule);
                }
            }

            // Create keyword arguments
            PyObject* pKwargs = PyDict_New();
            PyDict_SetItemString(
                pKwargs, "cst_display_type", PyUnicode_FromString(cst_display_type.c_str())
            );
            if (pPrimaries != Py_None) {
                PyDict_SetItemString(pKwargs, "display_primaries", pPrimaries);
            }
            Py_DECREF(pPrimaries);

            // Create positional arguments
            PyObject* pArgs = PyTuple_Pack(
                7,
                PyFloat_FromDouble(scramble_prob),
                PyUnicode_FromString(sex.c_str()),
                PyFloat_FromDouble(percentage_screened),
                PyFloat_FromDouble(peak_to_test),
                PyFloat_FromDouble(luminance),
                PyFloat_FromDouble(saturation),
                pDimensions
            );

            // Add seed to kwargs
            PyDict_SetItemString(pKwargs, "seed", PyLong_FromLong(seed));

            pInstance = reinterpret_cast<PyObject*>(
                PyObject_Call(reinterpret_cast<PyObject*>(pClass), pArgs, pKwargs)
            );
            Py_DECREF(pArgs);
            Py_DECREF(pKwargs);
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

std::vector<std::string> CircleGridGenerator::GetGenotypes() const
{
    std::vector<std::string> genotypes;
    if (!pInstance)
        return genotypes;

    PyObject* pMethod
        = PyObject_GetAttrString(reinterpret_cast<PyObject*>(pInstance), "GetGenotypes");
    if (!pMethod || !PyCallable_Check(pMethod)) {
        Py_XDECREF(pMethod);
        return genotypes;
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

std::vector<std::pair<int, int>> CircleGridGenerator::GetImages(
    const std::string& genotype,
    int metameric_axis,
    const std::vector<std::string>& filenames,
    ColorSpaceType output_space
)
{
    std::vector<std::pair<int, int>> idxs;
    if (!pInstance)
        return idxs;

    // Parse genotype string to tuple
    // Format is like "(558.9, 530.3)"
    PyObject* pGenotype = PyRun_String(genotype.c_str(), Py_eval_input, PyDict_New(), PyDict_New());
    if (!pGenotype) {
        PyErr_Print();
        return idxs;
    }

    PyObject* pList = PyList_New(filenames.size());
    for (size_t i = 0; i < filenames.size(); ++i) {
        PyList_SetItem(pList, i, PyUnicode_FromString(filenames[i].c_str()));
    }

    PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
    if (!pOutputSpace) {
        Py_DECREF(pGenotype);
        Py_DECREF(pList);
        return idxs;
    }

    PyObject* pValue = PyObject_CallMethod(
        reinterpret_cast<PyObject*>(pInstance),
        "GetImages",
        "OiOO",
        pGenotype,
        metameric_axis,
        pList,
        pOutputSpace
    );
    Py_DECREF(pGenotype);
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
