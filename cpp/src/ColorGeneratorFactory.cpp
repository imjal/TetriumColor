#include "TetriumColor/ColorGeneratorFactory.h"
#include <stdexcept>

namespace TetriumColor
{

PyObject* ColorGeneratorFactory::CreateQuestColorGenerator(
    const std::string& sex,
    float percentage_screened,
    float background_luminance,
    int trials_per_direction,
    const std::vector<int>& metameric_axes,
    const std::vector<int>& dimensions,
    const std::string& display_primaries_path,
    bool bipolar,
    float degree,
    int mcs_k
)
{
    // Load primaries from CSV first
    PyObject* pMeasurementModule = PyImport_ImportModule("TetriumColor.Measurement");
    if (!pMeasurementModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.Measurement module");
    }

    PyObject* pLoadPrimariesFunc
        = PyObject_GetAttrString(pMeasurementModule, "load_primaries_from_csv");
    Py_DECREF(pMeasurementModule);

    if (!pLoadPrimariesFunc) {
        PyErr_Print();
        throw std::runtime_error("Failed to get load_primaries_from_csv function");
    }

    PyObject* pPrimaries
        = PyObject_CallFunction(pLoadPrimariesFunc, "s", display_primaries_path.c_str());
    Py_DECREF(pLoadPrimariesFunc);

    if (!pPrimaries) {
        PyErr_Print();
        throw std::runtime_error("Failed to load primaries from CSV: " + display_primaries_path);
    }

    // Import the ColorGenerator module
    PyObject* pModule = PyImport_ImportModule("TetriumColor.TetraColorPicker");
    if (!pModule) {
        Py_DECREF(pPrimaries);
        PyErr_Print();
        throw std::runtime_error("Failed to import TetraColorPicker module");
    }

    // Get the QuestColorGenerator class
    PyObject* pClass = PyObject_GetAttrString(pModule, "QuestColorGenerator");
    Py_DECREF(pModule);

    if (!pClass) {
        Py_DECREF(pPrimaries);
        PyErr_Print();
        throw std::runtime_error("Failed to get QuestColorGenerator class");
    }

    // Build metameric_axes list
    // If empty, pass None so Python defaults to all axes (list(range(4)))
    // Otherwise, pass the list of specific axes to test
    PyObject* pMetamericAxes = nullptr;
    if (metameric_axes.empty()) {
        pMetamericAxes = Py_None; // None = test all axes (Python will default to list(range(4)))
        Py_INCREF(Py_None);       // Py_None is a singleton, but we need to increment ref for dict
    } else {
        pMetamericAxes = PyList_New(metameric_axes.size());
        for (size_t i = 0; i < metameric_axes.size(); i++) {
            PyList_SetItem(pMetamericAxes, i, PyLong_FromLong(metameric_axes[i]));
        }
    }

    // Build dimensions list
    PyObject* pDimensions = PyList_New(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); i++) {
        PyList_SetItem(pDimensions, i, PyLong_FromLong(dimensions[i]));
    }

    // Create instance with keyword arguments
    // QuestColorGenerator has many optional parameters, so we use PyObject_Call with kwargs dict
    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "luminance", PyFloat_FromDouble(background_luminance));
    PyDict_SetItemString(pKwargs, "trials_per_direction", PyLong_FromLong(trials_per_direction));
    PyDict_SetItemString(pKwargs, "dimensions", pDimensions);        // Steals reference
    PyDict_SetItemString(pKwargs, "metameric_axes", pMetamericAxes); // Steals reference
    PyDict_SetItemString(pKwargs, "display_primaries", pPrimaries);  // Steals reference
    PyDict_SetItemString(pKwargs, "bipolar", bipolar ? Py_True : Py_False);
    PyDict_SetItemString(pKwargs, "degree", PyFloat_FromDouble(degree));
    PyDict_SetItemString(pKwargs, "mcs_k", PyLong_FromLong(mcs_k));

    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(sex.c_str()));
    PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(percentage_screened));

    PyObject* pColorGenerator = PyObject_Call(pClass, pArgs, pKwargs);

    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    // Note: pDimensions reference was stolen by PyDict_SetItemString, don't DECREF it
    // Note: pPrimaries reference was stolen by PyDict_SetItemString, don't DECREF it
    // For pMetamericAxes: if it was Py_None, we incremented it so we need to decrement
    // If it was a list, the reference was stolen by PyDict_SetItemString
    if (metameric_axes.empty()) {
        Py_DECREF(pMetamericAxes); // Decrement the Py_None reference we incremented
    }
    Py_DECREF(pClass);

    if (!pColorGenerator) {
        PyErr_Print();
        throw std::runtime_error("Failed to create QuestColorGenerator instance");
    }

    return pColorGenerator;
}

PyObject* ColorGeneratorFactory::CreatePseudoIsochromaticPlateGenerator(
    PyObject* color_generator,
    int seed
)
{
    if (!color_generator) {
        throw std::runtime_error("ColorGenerator is null");
    }

    // Import the PlateGenerator module
    PyObject* pModule = PyImport_ImportModule("TetriumColor.TetraPlate");
    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetraPlate module");
    }

    // Get the PseudoIsochromaticPlateGenerator class
    PyObject* pClass = PyObject_GetAttrString(pModule, "PseudoIsochromaticPlateGenerator");
    Py_DECREF(pModule);

    if (!pClass) {
        PyErr_Print();
        throw std::runtime_error("Failed to get PseudoIsochromaticPlateGenerator class");
    }

    // Create instance with color_generator and seed
    PyObject* pTestGenerator = PyObject_CallFunction(pClass, "Oi", color_generator, seed);
    Py_DECREF(pClass);

    if (!pTestGenerator) {
        PyErr_Print();
        throw std::runtime_error("Failed to create PseudoIsochromaticPlateGenerator instance");
    }

    return pTestGenerator;
}

PyObject* ColorGeneratorFactory::CreateCircleGridGenerator(
    PyObject* color_generator,
    float scramble_prob,
    float luminance,
    float saturation
)
{
    if (!color_generator) {
        throw std::runtime_error("ColorGenerator is null");
    }

    // Import the PlateGenerator module
    PyObject* pModule = PyImport_ImportModule("TetriumColor.TetraPlate");
    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetraPlate module");
    }

    // Get the CircleGridGenerator class
    PyObject* pClass = PyObject_GetAttrString(pModule, "CircleGridGenerator");
    Py_DECREF(pModule);

    if (!pClass) {
        PyErr_Print();
        throw std::runtime_error("Failed to get CircleGridGenerator class");
    }

    // Create instance with color_generator and parameters
    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "scramble_prob", PyFloat_FromDouble(scramble_prob));
    PyDict_SetItemString(pKwargs, "luminance", PyFloat_FromDouble(luminance));
    PyDict_SetItemString(pKwargs, "saturation", PyFloat_FromDouble(saturation));

    PyObject* pArgs = PyTuple_New(1);
    Py_INCREF(color_generator); // PyTuple_SetItem steals reference, but we want to keep it
    PyTuple_SetItem(pArgs, 0, color_generator); // Steals reference

    PyObject* pTestGenerator = PyObject_Call(pClass, pArgs, pKwargs);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pClass);

    if (!pTestGenerator) {
        PyErr_Print();
        throw std::runtime_error("Failed to create CircleGridGenerator instance");
    }

    return pTestGenerator;
}

PyObject* ColorGeneratorFactory::CreateBipartiteFieldGenerator(
    PyObject* color_generator,
    int seed,
    int size
)
{
    if (!color_generator) {
        throw std::runtime_error("ColorGenerator is null");
    }

    PyObject* pModule = PyImport_ImportModule("TetriumColor.TetraPlate");
    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetraPlate module");
    }

    PyObject* pClass = PyObject_GetAttrString(pModule, "BipartiteFieldGenerator");
    Py_DECREF(pModule);

    if (!pClass) {
        PyErr_Print();
        throw std::runtime_error("Failed to get BipartiteFieldGenerator class");
    }

    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "seed", PyLong_FromLong(seed));
    PyDict_SetItemString(pKwargs, "size", PyLong_FromLong(size));

    PyObject* pArgs = PyTuple_New(1);
    Py_INCREF(color_generator);
    PyTuple_SetItem(pArgs, 0, color_generator);

    PyObject* pTestGenerator = PyObject_Call(pClass, pArgs, pKwargs);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pClass);

    if (!pTestGenerator) {
        PyErr_Print();
        throw std::runtime_error("Failed to create BipartiteFieldGenerator instance");
    }

    return pTestGenerator;
}

PyObject* ColorGeneratorFactory::CreateGaussianBlobGenerator(
    PyObject* color_generator,
    int seed,
    int size
)
{
    if (!color_generator) {
        throw std::runtime_error("ColorGenerator is null");
    }

    PyObject* pModule = PyImport_ImportModule("TetriumColor.TetraPlate");
    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetraPlate module");
    }

    PyObject* pClass = PyObject_GetAttrString(pModule, "GaussianBlobGenerator");
    Py_DECREF(pModule);

    if (!pClass) {
        PyErr_Print();
        throw std::runtime_error("Failed to get GaussianBlobGenerator class");
    }

    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "seed", PyLong_FromLong(seed));
    PyDict_SetItemString(pKwargs, "size", PyLong_FromLong(size));

    PyObject* pArgs = PyTuple_New(1);
    Py_INCREF(color_generator);
    PyTuple_SetItem(pArgs, 0, color_generator);

    PyObject* pTestGenerator = PyObject_Call(pClass, pArgs, pKwargs);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pClass);

    if (!pTestGenerator) {
        PyErr_Print();
        throw std::runtime_error("Failed to create GaussianBlobGenerator instance");
    }

    return pTestGenerator;
}

std::vector<float> ColorGeneratorFactory::GetObserverCDF(const std::string& sex, int dimension)
{
    PyObject* pModule = PyImport_ImportModule("TetriumColor.Observer.ObserverGenotypes");
    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to import TetriumColor.Observer.ObserverGenotypes");
    }

    PyObject* pClass = PyObject_GetAttrString(pModule, "ObserverGenotypes");
    Py_DECREF(pModule);
    if (!pClass) {
        PyErr_Print();
        throw std::runtime_error("Failed to get ObserverGenotypes class");
    }

    PyObject* pDimensions = PyList_New(1);
    PyList_SetItem(pDimensions, 0, PyLong_FromLong(dimension));

    PyObject* pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "dimensions", pDimensions);

    PyObject* pArgs = PyTuple_New(0);
    PyObject* pOG = PyObject_Call(pClass, pArgs, pKwargs);
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pClass);

    if (!pOG) {
        PyErr_Print();
        throw std::runtime_error("Failed to create ObserverGenotypes instance");
    }

    PyObject* pCDF = PyObject_CallMethod(pOG, "get_cdf", "s", sex.c_str());
    Py_DECREF(pOG);
    if (!pCDF) {
        PyErr_Print();
        throw std::runtime_error("Failed to call get_cdf");
    }

    PyObject* pValues = PyObject_CallMethod(pCDF, "values", nullptr);
    Py_DECREF(pCDF);
    if (!pValues) {
        PyErr_Print();
        throw std::runtime_error("Failed to get CDF values");
    }

    PyObject* pList = PySequence_List(pValues);
    Py_DECREF(pValues);
    if (!pList) {
        PyErr_Print();
        throw std::runtime_error("Failed to convert CDF values to list");
    }

    std::vector<float> result;
    Py_ssize_t n = PyList_Size(pList);
    result.reserve(n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* pVal = PyList_GetItem(pList, i); // borrowed reference
        result.push_back(static_cast<float>(PyFloat_AsDouble(pVal)));
    }
    Py_DECREF(pList);

    return result;
}

} // namespace TetriumColor
