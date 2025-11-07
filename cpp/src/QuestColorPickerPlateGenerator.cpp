#include "TetriumColor/QuestColorPickerPlateGenerator.h"
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

QuestColorPickerPlateGenerator::QuestColorPickerPlateGenerator(
    QuestColorPicker& color_picker,
    int seed
)
    : pModule(nullptr), pClass(nullptr), pInstance(nullptr), colorPicker(color_picker)
{
    // Import the TetriumColor.PsychoPhys.IshiharaPlate module for plate generation
    PyObject* pModuleName = PyUnicode_FromString("TetriumColor.PsychoPhys.IshiharaPlate");
    pModule = PyImport_Import(pModuleName);
    Py_DECREF(pModuleName);

    if (pModule == nullptr) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.PsychoPhys.IshiharaPlate module");
    }

    // Get the IshiharaPlateGenerator class
    pClass = PyObject_GetAttrString((PyObject*)pModule, "IshiharaPlateGenerator");
    if (pClass == nullptr || !PyCallable_Check((PyObject*)pClass)) {
        Py_XDECREF((PyObject*)pClass);
        Py_DECREF((PyObject*)pModule);
        throw std::runtime_error("Cannot find IshiharaPlateGenerator class");
    }

    // Create the plate generator instance (no parameters)
    PyObject* pArgs = PyTuple_New(0);
    pInstance = PyObject_CallObject((PyObject*)pClass, pArgs);
    Py_DECREF(pArgs);

    if (pInstance == nullptr) {
        PyErr_Print();
        Py_DECREF((PyObject*)pClass);
        Py_DECREF((PyObject*)pModule);
        throw std::runtime_error("Failed to create IshiharaPlateGenerator instance");
    }

    // Store the seed for use in GeneratePlate calls
    this->seed = seed;
}

QuestColorPickerPlateGenerator::~QuestColorPickerPlateGenerator()
{
    Py_XDECREF((PyObject*)pInstance);
    Py_XDECREF((PyObject*)pClass);
    Py_XDECREF((PyObject*)pModule);
}

void QuestColorPickerPlateGenerator::GetPlate(
    int direction_idx,
    const std::string& filename,
    int hidden_number,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise,
    double intensity
)
{
    // Get the genotype and metameric axis for this direction
    auto metadata = colorPicker.GetDirectionsMetadata();
    auto it = metadata.find(direction_idx);

    if (it == metadata.end()) {
        throw std::runtime_error("Direction index not found in metadata");
    }

    std::string genotype_str = it->second.first;
    int metameric_axis = it->second.second;

    // Get observer_genotypes and color_space from Quest picker
    PyObject* pQuestInstance = (PyObject*)colorPicker.GetPythonInstance();

    if (!pQuestInstance) {
        throw std::runtime_error("QuestColorPicker Python instance is NULL");
    }

    PyObject* pObserverGenotypes = PyObject_GetAttrString(pQuestInstance, "observer_genotypes");
    if (!pObserverGenotypes) {
        PyErr_Print();
        throw std::runtime_error("Cannot get observer_genotypes from QuestColorGenerator");
    }

    PyObject* pColorSpace = PyObject_GetAttrString(pQuestInstance, "color_space");
    if (!pColorSpace) {
        PyErr_Print();
        Py_DECREF(pObserverGenotypes);
        throw std::runtime_error("Cannot get color_space from QuestColorGenerator");
    }

    // Parse genotype string to tuple
    PyObject* pGlobals = PyDict_New();
    PyObject* pLocals = PyDict_New();
    PyObject* pGenotype = PyRun_String(genotype_str.c_str(), Py_eval_input, pGlobals, pLocals);
    Py_DECREF(pGlobals);
    Py_DECREF(pLocals);

    if (!pGenotype) {
        PyErr_Print();
        Py_DECREF(pObserverGenotypes);
        Py_DECREF(pColorSpace);
        throw std::runtime_error("Failed to parse genotype string");
    }

    // Get color space for this specific genotype using observer_genotypes
    PyObject* pGetColorSpaceMethod
        = PyObject_GetAttrString(pObserverGenotypes, "get_color_space_for_peaks");
    if (!pGetColorSpaceMethod) {
        Py_DECREF(pGenotype);
        Py_DECREF(pObserverGenotypes);
        Py_DECREF(pColorSpace);
        throw std::runtime_error("Cannot find get_color_space_for_peaks method");
    }

    // Call get_color_space_for_peaks(genotype, display_primaries=color_space.display_primaries)
    PyObject* pDisplayPrimaries = PyObject_GetAttrString(pColorSpace, "display_primaries");
    PyObject* pCSArgs = PyTuple_New(1);
    PyTuple_SetItem(pCSArgs, 0, pGenotype); // steals reference
    PyObject* pCSKwargs = PyDict_New();
    if (pDisplayPrimaries && pDisplayPrimaries != Py_None) {
        PyDict_SetItemString(pCSKwargs, "display_primaries", pDisplayPrimaries);
    }
    Py_XDECREF(pDisplayPrimaries);

    PyObject* pGenotypeColorSpace = PyObject_Call(pGetColorSpaceMethod, pCSArgs, pCSKwargs);
    Py_DECREF(pCSArgs);
    Py_DECREF(pCSKwargs);
    Py_DECREF(pGetColorSpaceMethod);
    Py_DECREF(pObserverGenotypes);
    Py_DECREF(pColorSpace);

    if (!pGenotypeColorSpace) {
        PyErr_Print();
        throw std::runtime_error("Failed to get color space for genotype");
    }

    // Get metameric pair using color_space.get_maximal_pair_in_disp_from_pt
    // First get a random point in DISP space
    PyObject* pNumpyModule = PyImport_ImportModule("numpy");
    if (!pNumpyModule) {
        PyErr_Print();
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to import numpy");
    }

    PyObject* pOnesFunc = PyObject_GetAttrString(pNumpyModule, "ones");
    if (!pOnesFunc) {
        PyErr_Print();
        Py_DECREF(pNumpyModule);
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to get numpy.ones");
    }

    PyObject* pDimAttr = PyObject_GetAttrString(pGenotypeColorSpace, "dim");
    if (!pDimAttr) {
        PyErr_Print();
        Py_DECREF(pOnesFunc);
        Py_DECREF(pNumpyModule);
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to get color_space.dim");
    }

    long dim = PyLong_AsLong(pDimAttr);
    Py_DECREF(pDimAttr);

    PyObject* pPointArgs = PyTuple_New(1);
    PyTuple_SetItem(pPointArgs, 0, PyLong_FromLong(dim));
    PyObject* pPoint = PyObject_CallObject(pOnesFunc, pPointArgs);
    Py_DECREF(pPointArgs);
    Py_DECREF(pOnesFunc);
    Py_DECREF(pNumpyModule);

    if (!pPoint) {
        PyErr_Print();
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to create point array");
    }

    // Multiply by 0.5 to get middle gray
    PyObject* pHalf = PyFloat_FromDouble(0.5);
    PyObject* pMultiplyResult = PyNumber_Multiply(pPoint, pHalf);
    Py_DECREF(pPoint);
    Py_DECREF(pHalf);

    if (!pMultiplyResult) {
        PyErr_Print();
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to multiply point by 0.5");
    }

    pPoint = pMultiplyResult;

    // Call get_maximal_pair_in_disp_from_pt
    PyObject* pGetPairMethod
        = PyObject_GetAttrString(pGenotypeColorSpace, "get_maximal_pair_in_disp_from_pt");
    if (!pGetPairMethod) {
        PyErr_Print();
        Py_DECREF(pPoint);
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to get get_maximal_pair_in_disp_from_pt method");
    }

    PyObject* pPairArgs = PyTuple_New(1);
    PyTuple_SetItem(pPairArgs, 0, pPoint); // steals reference
    PyObject* pPairKwargs = PyDict_New();
    PyDict_SetItemString(pPairKwargs, "metameric_axis", PyLong_FromLong(metameric_axis));
    PyDict_SetItemString(pPairKwargs, "proportion", PyFloat_FromDouble(intensity));

    PyObject* pPairResult = PyObject_Call(pGetPairMethod, pPairArgs, pPairKwargs);
    Py_DECREF(pPairArgs);
    Py_DECREF(pPairKwargs);
    Py_DECREF(pGetPairMethod);

    if (!pPairResult) {
        PyErr_Print();
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to call get_maximal_pair_in_disp_from_pt");
    }

    if (!PyTuple_Check(pPairResult) || PyTuple_Size(pPairResult) < 2) {
        Py_DECREF(pPairResult);
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("get_maximal_pair_in_disp_from_pt did not return a valid tuple");
    }

    PyObject* pInsideCone = PyTuple_GetItem(pPairResult, 0);  // borrowed
    PyObject* pOutsideCone = PyTuple_GetItem(pPairResult, 1); // borrowed
    Py_INCREF(pInsideCone);
    Py_INCREF(pOutsideCone);
    Py_DECREF(pPairResult);

    // Convert hidden number to Python object
    PyObject* pHiddenSymbol = PyLong_FromLong(hidden_number);
    PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);

    // Call IshiharaPlateGenerator.GeneratePlate
    PyObject* pGeneratePlateMethod = PyObject_GetAttrString((PyObject*)pInstance, "GeneratePlate");
    PyObject* pPlateArgs = PyTuple_New(5);
    PyTuple_SetItem(pPlateArgs, 0, pInsideCone);
    PyTuple_SetItem(pPlateArgs, 1, pOutsideCone);
    PyTuple_SetItem(pPlateArgs, 2, pGenotypeColorSpace);
    PyTuple_SetItem(pPlateArgs, 3, pHiddenSymbol);
    PyTuple_SetItem(pPlateArgs, 4, pOutputSpace);

    PyObject* pPlateKwargs = PyDict_New();
    PyDict_SetItemString(pPlateKwargs, "lum_noise", PyFloat_FromDouble(lum_noise));
    PyDict_SetItemString(pPlateKwargs, "s_cone_noise", PyFloat_FromDouble(s_cone_noise));
    PyDict_SetItemString(pPlateKwargs, "seed", PyLong_FromLong(seed));

    PyObject* pPlateResult = PyObject_Call(pGeneratePlateMethod, pPlateArgs, pPlateKwargs);
    Py_DECREF(pPlateArgs);
    Py_DECREF(pPlateKwargs);
    Py_DECREF(pGeneratePlateMethod);

    if (!pPlateResult) {
        PyErr_Print();
        throw std::runtime_error("Failed to generate plate");
    }

    // Export plate
    if (output_space == ColorSpaceType::DISP_6P) {
        PyObject* pExportMethod = PyObject_GetAttrString((PyObject*)pInstance, "ExportPlateTo6P");
        PyObject* pExportArgs = PyTuple_New(2);
        Py_INCREF(pPlateResult);
        PyTuple_SetItem(pExportArgs, 0, pPlateResult);
        PyTuple_SetItem(pExportArgs, 1, PyUnicode_FromString(filename.c_str()));
        PyObject* pExportResult = PyObject_CallObject(pExportMethod, pExportArgs);
        Py_DECREF(pExportArgs);
        Py_DECREF(pExportMethod);
        Py_XDECREF(pExportResult);
    } else {
        // Save SRGB image - pPlateResult should be a list with [image, ...]
        if (!PyList_Check(pPlateResult) || PyList_Size(pPlateResult) < 1) {
            Py_DECREF(pPlateResult);
            throw std::runtime_error("GeneratePlate did not return a valid list with image");
        }

        PyObject* pImage = PyList_GetItem(pPlateResult, 0); // borrowed reference
        if (!pImage) {
            PyErr_Print();
            Py_DECREF(pPlateResult);
            throw std::runtime_error("Failed to get image from plate result list");
        }

        PyObject* pSaveMethod = PyObject_GetAttrString(pImage, "save");
        if (!pSaveMethod) {
            PyErr_Print();
            Py_DECREF(pPlateResult);
            throw std::runtime_error("Image object does not have save method");
        }

        std::string srgb_filename = filename + "_SRGB.png";
        PyObject* pSaveArgs = PyTuple_New(1);
        PyTuple_SetItem(pSaveArgs, 0, PyUnicode_FromString(srgb_filename.c_str()));
        PyObject* pSaveResult = PyObject_CallObject(pSaveMethod, pSaveArgs);
        Py_DECREF(pSaveArgs);
        Py_DECREF(pSaveMethod);

        if (!pSaveResult) {
            PyErr_Print();
            Py_DECREF(pPlateResult);
            throw std::runtime_error("Failed to save image");
        }
        Py_DECREF(pSaveResult);
    }

    Py_DECREF(pPlateResult);
}

void QuestColorPickerPlateGenerator::GetPlate(
    int direction_idx,
    const std::string& filename,
    const std::string& hidden_symbol,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise,
    double intensity
)
{
    // Get the genotype and metameric axis for this direction
    auto metadata = colorPicker.GetDirectionsMetadata();
    auto it = metadata.find(direction_idx);

    if (it == metadata.end()) {
        throw std::runtime_error("Direction index not found in metadata");
    }

    std::string genotype_str = it->second.first;
    int metameric_axis = it->second.second;

    // Get observer_genotypes and color_space from Quest picker
    PyObject* pQuestInstance = (PyObject*)colorPicker.GetPythonInstance();

    if (!pQuestInstance) {
        throw std::runtime_error("QuestColorPicker Python instance is NULL");
    }

    PyObject* pObserverGenotypes = PyObject_GetAttrString(pQuestInstance, "observer_genotypes");
    if (!pObserverGenotypes) {
        PyErr_Print();
        throw std::runtime_error("Cannot get observer_genotypes from QuestColorGenerator");
    }

    PyObject* pColorSpace = PyObject_GetAttrString(pQuestInstance, "color_space");
    if (!pColorSpace) {
        PyErr_Print();
        Py_DECREF(pObserverGenotypes);
        throw std::runtime_error("Cannot get color_space from QuestColorGenerator");
    }

    // Parse genotype string to tuple
    PyObject* pGlobals = PyDict_New();
    PyObject* pLocals = PyDict_New();
    PyObject* pGenotype = PyRun_String(genotype_str.c_str(), Py_eval_input, pGlobals, pLocals);
    Py_DECREF(pGlobals);
    Py_DECREF(pLocals);

    if (!pGenotype) {
        PyErr_Print();
        Py_DECREF(pObserverGenotypes);
        Py_DECREF(pColorSpace);
        throw std::runtime_error("Failed to parse genotype string");
    }

    // Get color space for this specific genotype using observer_genotypes
    PyObject* pGetColorSpaceMethod
        = PyObject_GetAttrString(pObserverGenotypes, "get_color_space_for_peaks");
    if (!pGetColorSpaceMethod) {
        Py_DECREF(pGenotype);
        Py_DECREF(pObserverGenotypes);
        Py_DECREF(pColorSpace);
        throw std::runtime_error("Cannot find get_color_space_for_peaks method");
    }

    // Call get_color_space_for_peaks(genotype, display_primaries=color_space.display_primaries)
    PyObject* pDisplayPrimaries = PyObject_GetAttrString(pColorSpace, "display_primaries");
    PyObject* pCSArgs = PyTuple_New(1);
    PyTuple_SetItem(pCSArgs, 0, pGenotype); // steals reference
    PyObject* pCSKwargs = PyDict_New();
    if (pDisplayPrimaries && pDisplayPrimaries != Py_None) {
        PyDict_SetItemString(pCSKwargs, "display_primaries", pDisplayPrimaries);
    }
    Py_XDECREF(pDisplayPrimaries);

    PyObject* pGenotypeColorSpace = PyObject_Call(pGetColorSpaceMethod, pCSArgs, pCSKwargs);
    Py_DECREF(pCSArgs);
    Py_DECREF(pCSKwargs);
    Py_DECREF(pGetColorSpaceMethod);
    Py_DECREF(pObserverGenotypes);
    Py_DECREF(pColorSpace);

    if (!pGenotypeColorSpace) {
        PyErr_Print();
        throw std::runtime_error("Failed to get color space for genotype");
    }

    // Get metameric pair using color_space.get_maximal_pair_in_disp_from_pt
    // First get a random point in DISP space
    PyObject* pNumpyModule = PyImport_ImportModule("numpy");
    if (!pNumpyModule) {
        PyErr_Print();
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to import numpy");
    }

    PyObject* pOnesFunc = PyObject_GetAttrString(pNumpyModule, "ones");
    if (!pOnesFunc) {
        PyErr_Print();
        Py_DECREF(pNumpyModule);
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to get numpy.ones");
    }

    PyObject* pDimAttr = PyObject_GetAttrString(pGenotypeColorSpace, "dim");
    if (!pDimAttr) {
        PyErr_Print();
        Py_DECREF(pOnesFunc);
        Py_DECREF(pNumpyModule);
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to get color_space.dim");
    }

    long dim = PyLong_AsLong(pDimAttr);
    Py_DECREF(pDimAttr);

    PyObject* pPointArgs = PyTuple_New(1);
    PyTuple_SetItem(pPointArgs, 0, PyLong_FromLong(dim));
    PyObject* pPoint = PyObject_CallObject(pOnesFunc, pPointArgs);
    Py_DECREF(pPointArgs);
    Py_DECREF(pOnesFunc);
    Py_DECREF(pNumpyModule);

    if (!pPoint) {
        PyErr_Print();
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to create point array");
    }

    // Multiply by 0.5 to get middle gray
    PyObject* pHalf = PyFloat_FromDouble(0.5);
    PyObject* pMultiplyResult = PyNumber_Multiply(pPoint, pHalf);
    Py_DECREF(pPoint);
    Py_DECREF(pHalf);

    if (!pMultiplyResult) {
        PyErr_Print();
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to multiply point by 0.5");
    }

    pPoint = pMultiplyResult;

    // Call get_maximal_pair_in_disp_from_pt
    PyObject* pGetPairMethod
        = PyObject_GetAttrString(pGenotypeColorSpace, "get_maximal_pair_in_disp_from_pt");
    if (!pGetPairMethod) {
        PyErr_Print();
        Py_DECREF(pPoint);
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to get get_maximal_pair_in_disp_from_pt method");
    }

    PyObject* pPairArgs = PyTuple_New(1);
    PyTuple_SetItem(pPairArgs, 0, pPoint); // steals reference
    PyObject* pPairKwargs = PyDict_New();
    PyDict_SetItemString(pPairKwargs, "metameric_axis", PyLong_FromLong(metameric_axis));
    PyDict_SetItemString(pPairKwargs, "proportion", PyFloat_FromDouble(intensity));

    PyObject* pPairResult = PyObject_Call(pGetPairMethod, pPairArgs, pPairKwargs);
    Py_DECREF(pPairArgs);
    Py_DECREF(pPairKwargs);
    Py_DECREF(pGetPairMethod);

    if (!pPairResult) {
        PyErr_Print();
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("Failed to call get_maximal_pair_in_disp_from_pt");
    }

    if (!PyTuple_Check(pPairResult) || PyTuple_Size(pPairResult) < 2) {
        Py_DECREF(pPairResult);
        Py_DECREF(pGenotypeColorSpace);
        throw std::runtime_error("get_maximal_pair_in_disp_from_pt did not return a valid tuple");
    }

    PyObject* pInsideCone = PyTuple_GetItem(pPairResult, 0);  // borrowed
    PyObject* pOutsideCone = PyTuple_GetItem(pPairResult, 1); // borrowed
    Py_INCREF(pInsideCone);
    Py_INCREF(pOutsideCone);
    Py_DECREF(pPairResult);

    // Convert hidden symbol to Python string
    PyObject* pHiddenSymbol = PyUnicode_FromString(hidden_symbol.c_str());
    PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);

    // Call IshiharaPlateGenerator.GeneratePlate
    PyObject* pGeneratePlateMethod = PyObject_GetAttrString((PyObject*)pInstance, "GeneratePlate");
    PyObject* pPlateArgs = PyTuple_New(5);
    PyTuple_SetItem(pPlateArgs, 0, pInsideCone);
    PyTuple_SetItem(pPlateArgs, 1, pOutsideCone);
    PyTuple_SetItem(pPlateArgs, 2, pGenotypeColorSpace);
    PyTuple_SetItem(pPlateArgs, 3, pHiddenSymbol);
    PyTuple_SetItem(pPlateArgs, 4, pOutputSpace);

    PyObject* pPlateKwargs = PyDict_New();
    PyDict_SetItemString(pPlateKwargs, "lum_noise", PyFloat_FromDouble(lum_noise));
    PyDict_SetItemString(pPlateKwargs, "s_cone_noise", PyFloat_FromDouble(s_cone_noise));
    PyDict_SetItemString(pPlateKwargs, "seed", PyLong_FromLong(seed));

    PyObject* pPlateResult = PyObject_Call(pGeneratePlateMethod, pPlateArgs, pPlateKwargs);
    Py_DECREF(pPlateArgs);
    Py_DECREF(pPlateKwargs);
    Py_DECREF(pGeneratePlateMethod);

    if (!pPlateResult) {
        PyErr_Print();
        throw std::runtime_error("Failed to generate plate");
    }

    // Export plate
    if (output_space == ColorSpaceType::DISP_6P) {
        PyObject* pExportMethod = PyObject_GetAttrString((PyObject*)pInstance, "ExportPlateTo6P");
        PyObject* pExportArgs = PyTuple_New(2);
        Py_INCREF(pPlateResult);
        PyTuple_SetItem(pExportArgs, 0, pPlateResult);
        PyTuple_SetItem(pExportArgs, 1, PyUnicode_FromString(filename.c_str()));
        PyObject* pExportResult = PyObject_CallObject(pExportMethod, pExportArgs);
        Py_DECREF(pExportArgs);
        Py_DECREF(pExportMethod);
        Py_XDECREF(pExportResult);
    } else {
        // Save SRGB image - pPlateResult should be a list with [image, ...]
        if (!PyList_Check(pPlateResult) || PyList_Size(pPlateResult) < 1) {
            Py_DECREF(pPlateResult);
            throw std::runtime_error("GeneratePlate did not return a valid list with image");
        }

        PyObject* pImage = PyList_GetItem(pPlateResult, 0); // borrowed reference
        if (!pImage) {
            PyErr_Print();
            Py_DECREF(pPlateResult);
            throw std::runtime_error("Failed to get image from plate result list");
        }

        PyObject* pSaveMethod = PyObject_GetAttrString(pImage, "save");
        if (!pSaveMethod) {
            PyErr_Print();
            Py_DECREF(pPlateResult);
            throw std::runtime_error("Image object does not have save method");
        }

        std::string srgb_filename = filename + "_SRGB.png";
        PyObject* pSaveArgs = PyTuple_New(1);
        PyTuple_SetItem(pSaveArgs, 0, PyUnicode_FromString(srgb_filename.c_str()));
        PyObject* pSaveResult = PyObject_CallObject(pSaveMethod, pSaveArgs);
        Py_DECREF(pSaveArgs);
        Py_DECREF(pSaveMethod);

        if (!pSaveResult) {
            PyErr_Print();
            Py_DECREF(pPlateResult);
            throw std::runtime_error("Failed to save image");
        }
        Py_DECREF(pSaveResult);
    }

    Py_DECREF(pPlateResult);
}

} // namespace TetriumColor
