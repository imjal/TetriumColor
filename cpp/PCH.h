const static char* TETRIUM_COLOR_MODULE_PATH = "../extern/TetriumColor/";

/*
 * Workaround to compile python in debug without needing debug-version
 * of CPython https://discuss.python.org/t/cannot-open-file-python311-d-lib/32399/10
 */
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif