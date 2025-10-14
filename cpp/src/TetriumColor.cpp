#include "TetriumColor/TetriumColor.h"

namespace TetriumColor
{

void Init()
{
    Py_Initialize();
    // configure the Python path
    PyObject* sys_path = PyImport_ImportModule("sys");
    if (sys_path != nullptr) {
        PyObject* sys_path_obj = PyObject_GetAttrString(sys_path, "path");
        if (sys_path_obj != nullptr) {
            // Append the directory to the sys.path
            PyList_Append(sys_path_obj, PyUnicode_FromString(TETRIUM_COLOR_MODULE_PATH));
        }
    }
}

void Cleanup() { Py_Finalize(); }

} // namespace TetriumColor
