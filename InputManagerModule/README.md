# InputManager Module

This module provides a high-performance interface to capture and preprocess video frames from a camera. It is designed as a C++ backend with Python bindings, offering the speed of native code and the flexibility of Python for development.

## Demo

The `InputManager.py` script provides a demonstration of the module's capabilities. It captures a single frame and saves it as a NumPy array (`output.npy`), and also records a 5-second video (`output.mp4`).

To run the demo, execute the following command from the project's root directory:

```bash
python3 InputManager.py
```

This will generate the `output.npy` and `output.mp4` files in the root directory.

## 1. Architecture

The module consists of two main parts:

1.  **C++ Core (`InputManager.cpp`)**: A C++ class that uses the **OpenCV** library to handle the low-level details of opening a camera device, capturing frames, performing preprocessing, and managing resources.

2.  **Python Bindings (`Bindings.cpp`)**: A C++ file that uses the **pybind11** library to expose the C++ `InputManager` class to Python. It translates C++ objects and functions into a format Python can understand.

The general data flow is:
`Camera -> OpenCV (cv::Mat) -> Preprocessing (Resize, Normalize) -> pybind11 -> NumPy Array -> Python`

## 2. Why a Hybrid C++/Python System?

While Python is excellent for rapid development, real-time video processing is a task where performance is critical.

- **Performance**: C++ is a compiled language that runs directly on the hardware, offering significantly lower latency compared to interpreted Python. This is essential for achieving high frame rates and performing preprocessing steps without bottlenecks.

- **Direct Hardware Access**: The C++ core uses OpenCV's highly optimized, low-level APIs to interface directly with the camera hardware, providing efficient and robust access to the video stream.

- **Efficient Memory Handling**: The hybrid approach allows for careful memory management, preventing data from being needlessly copied between C++ and Python and ensuring there are no memory leaks.

## 3. Key Concepts in `Bindings.cpp`

### 3.1. Creating the Python Module

This is the main entry point that Python looks for when you run `import input_manager_native`.

```cpp
PYBIND11_MODULE(input_manager_native, m) {
    m.doc() = "Python bindings for the C++ InputManager";

    // Bindings for our class go here...
}
```

- **Why?** The `PYBIND11_MODULE` macro creates the boilerplate for a Python extension module. The `m` variable is our handle to the module, which we use to add classes and functions to it.

### 3.2. Binding the C++ Class and its Constructor

This code exposes our C++ `InputManager` class to Python, making it usable as if it were a native Python class.

```cpp
py::class_<InputManager>(m, "InputManager")
    .def(py::init<int>(),
         py::arg("camera_index") = 0,
         "Constructor, takes an optional camera index.");
```

- **Why?** `py::class_` creates the link between the C++ `InputManager` and the Python name `"InputManager"`. The `.def(py::init<>())` line specifically binds the C++ constructor, and `py::arg(...)` gives the Python argument a name and a default value, making the API more user-friendly.

### 3.3. The Most Important Part: Safe and Efficient Memory Management

When C++ creates an object (like a `cv::Mat` frame), Python's garbage collector doesn't know how to clean it up. If we are not careful, this leads to memory leaks. The `get_frame` binding solves this with a **`py::capsule`** that transfers ownership of the C++ memory to the Python object.

```cpp
.def("get_frame", [](InputManager &self) -> py::array_t<float> {
    // 1. Get the processed cv::Mat from C++
    cv::Mat frame = self.get_frame();
    
    // ... checks for empty frame ...

    // 2. Move the frame into a smart pointer for automatic management
    std::unique_ptr<cv::Mat> frame_ptr = std::make_unique<cv::Mat>(std::move(frame));
    
    // 3. Create a capsule that "boxes up" the C++ pointer and its deleter
    py::capsule frame_capsule(frame_ptr.release(), [](void *ptr) {
        std::unique_ptr<cv::Mat> deleter_ptr(static_cast<cv::Mat*>(ptr));
    });
    
    // 4. Create the NumPy array, giving it the capsule as its "base"
    return py::array_t<float>(
        {height, width, channels},
        {row_stride, col_stride, channel_stride},
        data_ptr,
        frame_capsule
    );
}, py::return_value_policy::take_ownership);
```

- **Why do we do this?**
  1.  We get a processed `cv::Mat` in C++.
  2.  We want to create a NumPy array that **shares the memory** of that `cv::Mat` to avoid a slow and unnecessary copy.
  3.  The `py::capsule` acts as a contract. We give it to the NumPy array as its memory "base".
  4.  When the NumPy array is no longer used in Python and gets garbage-collected, it sees the attached `capsule` and calls the lambda function we provided, which safely `delete`s the C++ `cv::Mat`.
  5.  This is the standard, safe, and efficient way to manage the lifetime of C++ objects that are owned by Python objects.

## 4. Frame Processing Pipeline

The `get_frame` function in `InputManager.cpp` performs the specific preprocessing steps required by many standard computer vision models.

1.  **Frame Capture**: Reads a frame from the camera, with built-in retry and reconnect logic in case of failure.
2.  **Mirroring**: Flips the image horizontally if enabled (default is true).
3.  **BGR to RGB Conversion**: Converts the 3-channel image from the BGR color space (OpenCV's default) to the RGB color space, which is standard for most neural networks.
4.  **Resizing**: Resizes the image to a fixed resolution of 224x224 pixels, the common input size for many models.
5.  **Type Conversion**: Converts the image from 8-bit integer (`CV_8UC3`) to 32-bit floating-point (`CV_32FC3`) and scales pixel values from the `[0, 255]` range to the `[0.0, 1.0]` range.
6.  **Normalization**: Normalizes the floating-point image by subtracting a mean and dividing by a standard deviation. This ensures the input data has a similar statistical distribution to the data the model was trained on. The default mean and std are `0.5`, but can be configured.

## 5. Python API Reference

The following methods are exposed to Python.

- `__init__(self, camera_index=0)`
  - **Description**: Initializes the InputManager.
  - **Args**: `camera_index` (int): The index of the camera to use. Defaults to `0`.

- `start_capture(self) -> bool`
  - **Description**: Opens the camera device and starts the capture stream.
  - **Returns**: `True` if the camera was started successfully, `False` otherwise.

- `stop_capture(self)`
  - **Description**: Stops the capture and releases the camera device.

- `is_capture_open(self) -> bool`
  - **Description**: Checks if the camera capture is currently open and running.
  - **Returns**: `True` if the camera is open, `False` otherwise.

- `get_frame(self) -> numpy.ndarray`
  - **Description**: Captures, processes, and returns the current frame from the camera. The processing pipeline includes resizing to 224x224, BGR-to-RGB conversion, and normalization.
  - **Returns**: A 3D NumPy array of shape `(224, 224, 3)` and `float32` data type. Returns an empty array if a frame could not be captured.

- `set_mirror_enabled(self, enabled: bool)`
  - **Description**: Enables or disables the horizontal mirroring of the camera feed.
  - **Args**: `enabled` (bool): `True` to enable mirroring (default), `False` to disable it.

- `set_normalization_params(self, mean: float, std: float)`
  - **Description**: Sets the mean and standard deviation values used for frame normalization.
  - **Args**:
    - `mean` (float): The mean value to subtract from each pixel channel.
    - `std` (float): The standard deviation value to divide each pixel channel by.

## 6. How to Build

The module is built using CMake. Run the following commands from the project root directory:

```bash
mkdir -p build
cd build
cmake ..
make
```

This will generate the `input_manager_native.cpython-*.so` file (or `.pyd` on Windows) in the `build/lib` directory.

## 7. Running the Demo

The project includes a demonstration script, `InputManager.py`, which tests if the module works. The script performs two main actions:

1.  **Capture a single frame**: It captures one frame from the camera and saves it as a NumPy array to `output.npy`.
2.  **Record a video**: It records a 5-second video and saves it as `output.mp4`.

To run the demo, execute the following command from the project's root directory:

```bash
python3 InputManager.py
```

You should see output indicating the success of these actions, and the `output.npy` and `output.mp4` files will be created in the project's root directory.
