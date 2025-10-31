cmake_minimum_required(VERSION 3.22)

# Define the project name and version
project(TetriumColor VERSION 1.0 LANGUAGES CXX)

# ENV CONDA_PREFIX must be set to the conda environment path
if (NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "CONDA_PREFIX environment variable not set.\n Likely you have not activated the tetrium environment.")
endif()

message("Using conda environment at $ENV{CONDA_PREFIX}")

# Force CMake to find Python from conda environment ONLY
set(Python3_ROOT_DIR $ENV{CONDA_PREFIX})
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_REGISTRY NEVER)
set(Python3_FIND_VIRTUALENV FIRST)

# On macOS, prevent finding Homebrew Framework Python
if(APPLE)
    set(Python3_FIND_FRAMEWORK NEVER)  # Critical for macOS - prevents finding Homebrew Python
    message(STATUS "macOS detected - disabling Framework search to use conda Python")
elseif(WIN32)
    set(Python3_ROOT_DIR "C:/Users/yty10/miniconda3/envs/tetrium")
endif()

# Find Python libraries - THIS MUST COME BEFORE manually setting paths
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Verify we found the right Python
message(STATUS "Found Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Found Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Found Python3_LIBRARIES: ${Python3_LIBRARIES}")

# Check if we got the conda Python

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source files
set(SOURCES
    cpp/src/ColorGenerator.cpp
    cpp/src/GeneticColorPicker.cpp
    cpp/src/GeneticColorPickerPlateGenerator.cpp
    cpp/src/PseudoIsochromaticPlateGenerator.cpp
    cpp/src/TetriumColor.cpp
    cpp/src/CircleGridGenerator.cpp
    cpp/src/SolidColorGenerator.cpp
)

# Create the library
add_library(TetriumColor STATIC ${SOURCES})

# Add the include directory
target_include_directories(TetriumColor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${Python3_INCLUDE_DIRS}
)

# Link Python libraries
target_link_libraries(TetriumColor ${Python3_LIBRARIES})

target_precompile_headers(${PROJECT_NAME} PRIVATE cpp/PCH.h)