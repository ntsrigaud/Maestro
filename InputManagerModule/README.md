# InputManager Module

This module provides a high-performance interface to capture video frames from a camera. It is designed as a C++ backend with Python bindings, offering the speed of native code and the flexibility of Python for development.

## Architecture

The module consists of two main parts:

1.  **C++ Core (`InputManager.cpp`)**: A C++ class that uses the **OpenCV** library to handle the low-level details of opening a camera device, capturing frames, and managing resources.

2.  **Python Bindings (`Bindings.cpp`)**: A C++ file that uses the **pybind11** library to expose the C++ `InputManager` class to Python. It translates C++ objects and functions into a format Python can understand.

The general data flow is:
`Camera -> OpenCV (cv::Mat) -> pybind11 -> NumPy Array -> Python`

## Why a Hybrid C++/Python System?

While Python is excellent for rapid development, real-time video capture is a task where performance is critical.

- **Performance**: C++ is a compiled language that runs directly on the hardware, offering significantly lower latency compared to interpreted Python. This is essential for achieving high frame rates and avoiding dropped frames.

- **Direct Hardware Access**: The C++ core uses OpenCV's highly optimized, low-level APIs to interface directly with the camera hardware, providing efficient and robust access to the video stream.

- **Efficient Memory Handling**: As explained below, this hybrid approach allows for careful memory management, preventing data from being needlessly copied between C++ and Python and ensuring there are no memory leaks.

## Key Concepts in `Bindings.cpp`

### 1. Creating the Python Module

This is the main entry point that Python looks for when you run `import input_manager_native`.

```cpp
PYBIND11_MODULE(input_manager_native, m) {
    m.doc() = "Python bindings for the C++ InputManager";

    // Bindings for our class goes here...
}
```

- **Why?** The `PYBIND11_MODULE` macro creates the boilerplate for a Python extension module. The `m` variable is our handle to the module, which we use to add classes and functions to it.

### 2. Binding the C++ Class and its Constructor

This code exposes our C++ `InputManager` class to Python, making it usable as if it were a native Python class.

```cpp
py::class_<InputManager>(m, "InputManager")
    .def(py::init<int>(),
         py::arg("camera_index") = 0,
         "Constructor, takes an optional camera index.");
```

- **Why?** `py::class_` creates the link between the C++ `InputManager` and the Python name `"InputManager"`. The `.def(py::init<>())` line specifically binds the C++ constructor, and `py::arg(...)` gives the Python argument a name and a default value, making the API more user-friendly.

### 3. The Most Important Part: Safe Memory Management

When C++ creates an object (like a `cv::Mat` frame), Python's garbage collector doesn't know how to clean it up. If we are not careful, this leads to memory leaks.

The `get_frame` binding solves this with a **`py::capsule`**.

```cpp
// 1. A custom deleter function for our C++ object
void mat_deleter(void *mat_ptr) {
    delete static_cast<cv::Mat *>(mat_ptr);
}

// 2. Inside the get_frame binding...
.def("get_frame", [](InputManager &self) -> py::array_t<unsigned char> {
    // ... logic to get the cv::Mat frame ...

    // Create a pointer to the C++ object we need to manage
    cv::Mat *frame_ptr = output_frame.release();

    // 3. Create a capsule that "boxes up" the C++ pointer and our deleter
    py::capsule capsule(frame_ptr, mat_deleter);

    // 4. Create the NumPy array, giving it the capsule as its "base"
    return py::array_t<unsigned char>(
        shape,
        strides,
        frame_ptr->data, // Pointer to the raw image data
        capsule          // The memory manager
    );
}, py::return_value_policy::take_ownership);
```

- **Why do we do this?**
  1.  We create a `cv::Mat` in C++.
  2.  We want to create a NumPy array that **shares the memory** of that `cv::Mat` to avoid a slow and unnecessary copy.
  3.  The `py::capsule` acts as a contract. We give it to the NumPy array.
  4.  When the NumPy array is no longer used in Python and gets garbage-collected, it sees the attached `capsule` and says, "Ah, I need to run the special cleanup function you gave me." It then calls our `mat_deleter`, which safely `delete`s the C++ `cv::Mat`.
  5.  This is the standard, safe, and efficient way to manage the lifetime of C++ objects that are owned by Python objects.

## How to Build

The module is built using CMake. Run the following commands from the project root directory:

```bash
mkdir -p build
cd build
cmake ..
make
```

This will generate the `input_manager_native.cpython-*.so` file (or `.pyd` on Windows).

## Running the Demo

The project includes a demonstration script, `InputManager.py`, which tests the core functionality of the native module.

The script performs two main tests:

1.  It attempts to initialize, start, and capture a frame from a valid camera index (e.g., 0).
2.  It attempts to initialize a camera with an invalid index (e.g., 99) to ensure errors are handled correctly.

To run the demo, execute the following command from the project's root directory:

```bash
python3 InputManager.py
```

You should see output indicating the success or failure of these tests, confirming that the native module is working as expected.
