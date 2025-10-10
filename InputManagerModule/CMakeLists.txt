cmake_minimum_required(VERSION 4.1.1)

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif ()

project(InputManager)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED COMPONENTS core videoio imgproc)
find_package(pybind11 CONFIG REQUIRED)

set(SOURCES src/InputManager.cpp src/Bindings.cpp)

pybind11_add_module(input_manager_native ${SOURCES})

target_link_libraries(input_manager_native PRIVATE ${OpenCV_LIBS} pybind11::module)
target_include_directories(input_manager_native PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_DIR})

set_target_properties(input_manager_native PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

if (WIN32)
    set_target_properties(input_manager_native PROPERTIES SUFFIX ".pyd")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(input_manager_native PRIVATE -O3)
endif ()

if (MSVC)
    add_compile_options(/W4)
else ()
    add_compile_options(-Wall -Wextra)
endif ()
