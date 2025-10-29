#include "TetriumColor/SolidColorGenerator.h"
#include "TetriumColor/ColorSpaceType.h"
#include <Python.h>
#include <cstdio>

namespace TetriumColor
{

SolidColorGenerator::SolidColorGenerator(const std::string& primary_path)
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
    PyObject* pPrimaries = PyObject_CallFunction(pLoadPrimariesFunc, "s", primary_path.c_str());
    Py_DECREF(pLoadPrimariesFunc);
    Py_DECREF(pMeasurementModule);

    if (!pPrimaries) {
        PyErr_Print();
        printf("Failed to load primaries from %s\n", primary_path.c_str());
        exit(-1);
    }

    // Import Observer module
    PyObject* pObserverModule = PyImport_ImportModule("TetriumColor.Observer");
    if (!pObserverModule) {
        PyErr_Print();
        exit(-1);
    }

    // Get Observer class and create tetrachromat observer
    PyObject* pObserverClass = PyObject_GetAttrString(pObserverModule, "Observer");
    PyObject* pTetrachromatMethod = PyObject_GetAttrString(pObserverClass, "tetrachromat");
    PyObject* pObserver = PyObject_CallObject(pTetrachromatMethod, nullptr);

    Py_DECREF(pTetrachromatMethod);
    Py_DECREF(pObserverClass);
    Py_DECREF(pObserverModule);

    if (!pObserver) {
        PyErr_Print();
        exit(-1);
    }

    // Import ColorSpace module
    PyObject* pColorSpaceModule = PyImport_ImportModule("TetriumColor.ColorSpace");
    if (!pColorSpaceModule) {
        PyErr_Print();
        exit(-1);
    }

    // Create ColorSpace with observer, display_primaries, and cst_display_type
    PyObject* pColorSpaceClass = PyObject_GetAttrString(pColorSpaceModule, "ColorSpace");
    Py_DECREF(pColorSpaceModule);

    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pObserver); // pObserver reference stolen

    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "display_primaries", pPrimaries);
    PyDict_SetItemString(pKwargs, "cst_display_type", PyUnicode_FromString("led"));

    PyObject* pColorSpace = PyObject_Call(pColorSpaceClass, pArgs, pKwargs);

    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pColorSpaceClass);
    Py_DECREF(pPrimaries);

    if (!pColorSpace) {
        PyErr_Print();
        exit(-1);
    }

    // Import SolidColorGenerator module
    PyObject* pName = PyUnicode_DecodeFSDefault("TetriumColor.SolidColorGenerator");
    pModule = reinterpret_cast<PyObject*>(PyImport_Import(pName));
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the SolidColorGenerator class
        pClass = reinterpret_cast<PyObject*>(
            PyObject_GetAttrString(reinterpret_cast<PyObject*>(pModule), "SolidColorGenerator")
        );

        if (pClass && PyCallable_Check(reinterpret_cast<PyObject*>(pClass)) == 1) {
            // Create instance with color_space
            PyObject* pInstanceArgs = PyTuple_Pack(1, pColorSpace);
            pInstance = reinterpret_cast<PyObject*>(
                PyObject_CallObject(reinterpret_cast<PyObject*>(pClass), pInstanceArgs)
            );
            Py_DECREF(pInstanceArgs);

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

    Py_DECREF(pColorSpace);
}

SolidColorGenerator::~SolidColorGenerator()
{
    Py_XDECREF(pInstance);
    Py_XDECREF(pClass);
    Py_XDECREF(pModule);
}

std::pair<std::string, std::string> SolidColorGenerator::GenerateCircle(
    const std::string& filename_base,
    float r,
    float g,
    float b,
    float o,
    int image_size,
    float circle_radius_ratio,
    bool has_noisy_boundary,
    ColorSpaceType output_space
)
{
    if (pInstance == nullptr) {
        printf("SolidColorGenerator instance is null\n");
        return {"", ""};
    }

    // Get ColorSpaceType enum value
    PyObject* pColorSpaceTypeModule = PyImport_ImportModule("TetriumColor.ColorSpace");
    if (!pColorSpaceTypeModule) {
        PyErr_Print();
        return {"", ""};
    }

    PyObject* pColorSpaceTypeClass
        = PyObject_GetAttrString(pColorSpaceTypeModule, "ColorSpaceType");
    if (!pColorSpaceTypeClass) {
        PyErr_Print();
        Py_DECREF(pColorSpaceTypeModule);
        return {"", ""};
    }

    PyObject* pOutputSpace = nullptr;
    switch (output_space) {
    case ColorSpaceType::SRGB:
        pOutputSpace = PyObject_GetAttrString(pColorSpaceTypeClass, "SRGB");
        break;
    case ColorSpaceType::DISP_6P:
        pOutputSpace = PyObject_GetAttrString(pColorSpaceTypeClass, "DISP_6P");
        break;
    default:
        pOutputSpace = PyObject_GetAttrString(pColorSpaceTypeClass, "DISP_6P");
        break;
    }

    Py_DECREF(pColorSpaceTypeClass);
    Py_DECREF(pColorSpaceTypeModule);

    if (!pOutputSpace) {
        PyErr_Print();
        return {"", ""};
    }

    // Build RGBO tuple - SetItem steals the reference, so no need to DECREF the floats
    PyObject* pRGBOTuple = PyTuple_New(4);
    PyTuple_SetItem(pRGBOTuple, 0, PyFloat_FromDouble(r));
    PyTuple_SetItem(pRGBOTuple, 1, PyFloat_FromDouble(g));
    PyTuple_SetItem(pRGBOTuple, 2, PyFloat_FromDouble(b));
    PyTuple_SetItem(pRGBOTuple, 3, PyFloat_FromDouble(o));

    // Call the method - pass tuple as object
    // Format: s=string, O=object, i=int, f=float, i=int (for bool), O=object
    PyObject* pResult = PyObject_CallMethod(
        reinterpret_cast<PyObject*>(pInstance),
        "generate_circle",
        "sOifiO",
        filename_base.c_str(),
        pRGBOTuple,
        image_size,
        circle_radius_ratio,
        has_noisy_boundary ? 1 : 0, // Pass as int (Python will convert to bool)
        pOutputSpace
    );

    Py_DECREF(pRGBOTuple);
    Py_DECREF(pOutputSpace);

    if (!pResult) {
        PyErr_Print();
        printf("Failed to call generate_circle method\n");
        return {"", ""};
    }

    // Check if result is a tuple
    if (!PyTuple_Check(pResult)) {
        printf("generate_circle did not return a tuple\n");
        Py_DECREF(pResult);
        return {"", ""};
    }

    if (PyTuple_Size(pResult) != 2) {
        printf("generate_circle returned wrong tuple size: %ld\n", PyTuple_Size(pResult));
        Py_DECREF(pResult);
        return {"", ""};
    }

    // Extract the returned tuple (rgb_path, ocv_path)
    PyObject* pRGBPath = PyTuple_GetItem(pResult, 0); // Borrowed reference
    PyObject* pOCVPath = PyTuple_GetItem(pResult, 1); // Borrowed reference

    if (!pRGBPath || !pOCVPath) {
        printf("Failed to extract paths from tuple\n");
        Py_DECREF(pResult);
        return {"", ""};
    }

    const char* rgb_cstr = PyUnicode_AsUTF8(pRGBPath);
    const char* ocv_cstr = PyUnicode_AsUTF8(pOCVPath);

    if (!rgb_cstr || !ocv_cstr) {
        printf("Failed to convert paths to strings\n");
        Py_DECREF(pResult);
        return {"", ""};
    }

    std::string rgb_path = rgb_cstr;
    std::string ocv_path = ocv_cstr;

    printf("Generated textures: %s, %s\n", rgb_path.c_str(), ocv_path.c_str());

    Py_DECREF(pResult);

    return {rgb_path, ocv_path};
}

} // namespace TetriumColor
