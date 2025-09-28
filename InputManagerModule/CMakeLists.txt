cmake_minimum_required(VERSION 4.1.1)
project(InputManager)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    add_compile_options(/W4)
else ()
    add_compile_options(-Wall -Wextra)
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED COMPONENTS core videoio imgproc)

if (OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else ()
    message(
        FATAL_ERROR
            "OpenCV not found. Install it via:
        Windows: vcpkg install opencv
        macOS: brew install opencv
        Linux: sudo apt-get install libopencv-dev")
endif ()

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.5
    QUIET)
FetchContent_MakeAvailable(pybind11)

if (WIN32)
    set_target_properties(input_manager_native PROPERTIES SUFFIX ".pyd")
endif ()

pybind11_add_module(input_manager_native InputManager.cpp Bindings.cpp)

target_link_libraries(input_manager_native PRIVATE ${OpenCV_LIBS})
target_include_directories(input_manager_native PRIVATE ${OpenCV_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(input_manager_native PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(input_manager_native PRIVATE -O3)
endif ()
