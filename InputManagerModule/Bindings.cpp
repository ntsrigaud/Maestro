#include "InputManager.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(input_manager_native, m) {
    m.doc() = "Python bindings for the C++ InputManager";

    py::class_<InputManager>(m, "InputManager")
        .def(py::init<int>(),
             py::arg("camera_index") = 0,
             "Constructor, takes an optional camera index.")
        .def("start_capture", &InputManager::start_capture,
             "Starts the camera capture. Returns True on success.")
        .def("stop_capture", &InputManager::stop_capture,
             "Stops the camera capture and releases the device.")
        .def("is_capture_open", &InputManager::is_capture_open,
             "Returns True if the camera capture is open and running.")
        .def("set_normalization_params", &InputManager::set_normalization_params,
             "Set normalization parameters for the frame processing.")
        .def("set_mirror_enabled", &InputManager::set_mirror_enabled,
             "Enable or disable mirror effect.")
        .def("get_frame", [](InputManager &self) -> py::array_t<float> {
            cv::Mat frame = self.get_frame();
            
            if (frame.empty()) {
                return py::array_t<float>(); 
            }
            
            if (!frame.isContinuous()) {
                frame = frame.clone();
            }
            
            const ssize_t height = frame.rows;
            const ssize_t width = frame.cols;
            const ssize_t channels = frame.channels();
            
            std::unique_ptr<cv::Mat> frame_ptr = std::make_unique<cv::Mat>(std::move(frame));
            float* data_ptr = frame_ptr->ptr<float>();
            
            ssize_t row_stride = width * channels * sizeof(float);
            ssize_t col_stride = channels * sizeof(float);
            ssize_t channel_stride = sizeof(float);
            
            py::capsule frame_capsule(frame_ptr.release(), [](void *ptr) {
                std::unique_ptr<cv::Mat> deleter_ptr(static_cast<cv::Mat*>(ptr));
            });
            
            return py::array_t<float>(
                {height, width, channels},
                {row_stride, col_stride, channel_stride},
                data_ptr,
                frame_capsule
            ); }, py::return_value_policy::take_ownership, "Gets the current frame from the camera as a NumPy array.\n"
                                                                                                                     "Returns an empty array if no frame is available.");

    m.attr("__version__") = "1.0.0";
}
