#include "TetriumColor/PseudoIsochromaticPlateGenerator.h"
#include <vector>

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

PseudoIsochromaticPlateGenerator::PseudoIsochromaticPlateGenerator(
    ColorGenerator& color_generator,
    int seed
)
{
    // Import the Python module
    PyObject* pName = PyUnicode_DecodeFSDefault("TetriumColor.TetraPlate");
    pModule = reinterpret_cast<PyObject*>(PyImport_Import(pName));
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the Python class
        pClass = reinterpret_cast<PyObject*>(PyObject_GetAttrString(
            reinterpret_cast<PyObject*>(pModule), "PseudoIsochromaticPlateGenerator"
        ));

        if (pClass && PyCallable_Check(reinterpret_cast<PyObject*>(pClass)) == 1) {
            // Create an instance of the Python class with color_generator and seed
            PyObject* pArgs = PyTuple_Pack(
                2,
                color_generator.pInstance, // Pass the color generator instance
                PyLong_FromLong(seed)
            );
            pInstance = reinterpret_cast<PyObject*>(
                PyObject_CallObject(reinterpret_cast<PyObject*>(pClass), pArgs)
            );
            Py_DECREF(pArgs);

            if (!pInstance) {
                PyErr_Print();
                exit(-1);
            }
            // Grab default color space from color_generator.color_spaces[0]
            PyObject* pColorSpaces = PyObject_GetAttrString(
                reinterpret_cast<PyObject*>(color_generator.pInstance), "color_spaces"
            );
            if (!pColorSpaces) {
                PyErr_Print();
                exit(-1);
            }
            PyObject* pFirst = PyList_GetItem(pColorSpaces, 0); // borrowed ref
            if (!pFirst) {
                PyErr_Print();
                Py_DECREF(pColorSpaces);
                exit(-1);
            }
            Py_INCREF(pFirst); // store our own reference
            pDefaultColorSpace = pFirst;
            Py_DECREF(pColorSpaces);
        } else {
            PyErr_Print();
            exit(-1);
        }
    } else {
        PyErr_Print();
        exit(-1);
    }
}

PseudoIsochromaticPlateGenerator::~PseudoIsochromaticPlateGenerator()
{
    Py_XDECREF(pInstance);
    Py_XDECREF(pClass);
    Py_XDECREF(pModule);
    Py_XDECREF(pDefaultColorSpace);
}

void PseudoIsochromaticPlateGenerator::NewPlate(
    const std::string& filename,
    int hidden_number,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise
)
{
    if (pInstance != nullptr) {
        PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
        if (!pOutputSpace) {
            printf("Failed to convert ColorSpaceType to Python\n");
            return;
        }

        PyObject* pValue = PyObject_CallMethod(
            reinterpret_cast<PyObject*>(pInstance),
            "NewPlate",
            "siOff",
            filename.c_str(),
            hidden_number,
            pOutputSpace,
            lum_noise,
            s_cone_noise
        );

        Py_DECREF(pOutputSpace);

        if (pValue != nullptr) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    } else {
        printf("PseudoIsochromaticPlateGenerator instance is null\n");
        exit(-1);
    }
}

void PseudoIsochromaticPlateGenerator::NewPlate(
    const std::string& filename,
    const std::string& hidden_symbol,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise
)
{
    if (pInstance != nullptr) {
        PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
        if (!pOutputSpace) {
            printf("Failed to convert ColorSpaceType to Python\n");
            return;
        }

        PyObject* pValue = PyObject_CallMethod(
            reinterpret_cast<PyObject*>(pInstance),
            "NewPlate",
            "ssOff",
            filename.c_str(),
            hidden_symbol.c_str(),
            pOutputSpace,
            lum_noise,
            s_cone_noise
        );

        Py_DECREF(pOutputSpace);

        if (pValue != nullptr) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    } else {
        printf("PseudoIsochromaticPlateGenerator instance is null\n");
        exit(-1);
    }
}

void PseudoIsochromaticPlateGenerator::GetPlate(
    ColorTestResult previous_result,
    const std::string& filename,
    const std::string& hidden_symbol,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise
)
{
    if (pInstance != nullptr) {
        PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
        if (!pOutputSpace) {
            printf("Failed to convert ColorSpaceType to Python\n");
            return;
        }

        // Convert ColorTestResult to Python
        PyObject* pResult = PyLong_FromLong(static_cast<int>(previous_result));

        PyObject* pValue = PyObject_CallMethod(
            reinterpret_cast<PyObject*>(pInstance),
            "GetPlate",
            "OssOff",
            pResult,
            filename.c_str(),
            hidden_symbol.c_str(),
            pOutputSpace,
            lum_noise,
            s_cone_noise
        );

        Py_DECREF(pOutputSpace);
        Py_DECREF(pResult);

        if (pValue != nullptr) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    } else {
        printf("PseudoIsochromaticPlateGenerator instance is null\n");
        exit(-1);
    }
}

void PseudoIsochromaticPlateGenerator::GetLuminancePlate(
    const std::string& filename,
    const std::string& hidden_symbol,
    ColorSpaceType output_space,
    float lum_noise,
    float s_cone_noise
)
{
    if (pInstance != nullptr) {
        PyObject* pOutputSpace = ColorSpaceTypeToPython(output_space);
        if (!pOutputSpace) {
            printf("Failed to convert ColorSpaceType to Python\n");
            return;
        }

        if (!pDefaultColorSpace) {
            printf("Default ColorSpace is null\n");
            Py_DECREF(pOutputSpace);
            exit(-1);
        }

        PyObject* pValue = PyObject_CallMethod(
            reinterpret_cast<PyObject*>(pInstance),
            "GetLuminancePlate",
            "ssOffO",
            filename.c_str(),
            hidden_symbol.c_str(),
            reinterpret_cast<PyObject*>(pDefaultColorSpace),
            lum_noise,
            s_cone_noise,
            pOutputSpace
        );

        Py_DECREF(pOutputSpace);

        if (pValue != nullptr) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    } else {
        printf("PseudoIsochromaticPlateGenerator instance is null\n");
        exit(-1);
    }
}

} // namespace TetriumColor
