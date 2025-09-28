#include "InputManager.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(input_manager_native, m) {
    m.doc() = "Python bindings for the C++ InputManager";

    py::class_<FrameData>(m, "FrameData")
        .def(py::init<>())
        .def_readwrite("timestamp", &FrameData::timestamp)
        .def_readwrite("frame", &FrameData::frame)
        .def("get_frame", [](FrameData &self) -> py::array_t<float> {
            if (self.frame.empty()) {
                return py::array_t<float>();
            }
            
            if (!self.frame.isContinuous()) {
                self.frame = self.frame.clone();
            }
            
            const ssize_t height = self.frame.rows;
            const ssize_t width = self.frame.cols;
            const ssize_t channels = self.frame.channels();
            
            float* data_ptr = self.frame.ptr<float>();
            
            ssize_t row_stride = width * channels * sizeof(float);
            ssize_t col_stride = channels * sizeof(float);
            ssize_t channel_stride = sizeof(float);
            
            py::capsule capsule = py::capsule(data_ptr, [](void *) { });
            
            return py::array_t<float>(
                {height, width, channels},
                {row_stride, col_stride, channel_stride},
                data_ptr,
                capsule
            ); }, py::return_value_policy::take_ownership);

    py::class_<InputManager>(m, "InputManager")
        .def(py::init<int>(), py::arg("camera_index") = 0)
        .def("start_capture", &InputManager::start_capture)
        .def("stop_capture", &InputManager::stop_capture)
        .def("is_capture_open", &InputManager::is_capture_open)
        .def("start_continuous_capture", &InputManager::start_continuous_capture)
        .def("stop_continuous_capture", &InputManager::stop_continuous_capture)
        .def("get_latest_frame", &InputManager::get_latest_frame)
        .def("get_all_frames", &InputManager::get_all_frames)
        .def("set_normalization_params", &InputManager::set_normalization_params)
        .def("set_mirror_enabled", &InputManager::set_mirror_enabled);

    m.attr("__version__") = "1.0.0";
}
