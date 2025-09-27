#include "InputManager.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void mat_deleter(void *mat_ptr) {
    delete static_cast<cv::Mat *>(mat_ptr);
}

struct MatDeleter {
    void operator()(cv::Mat *mat) const {
        delete mat;
    }
};

using MatPtr = std::unique_ptr<cv::Mat, MatDeleter>;

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
        .def("get_frame", [](InputManager &self) -> py::array_t<unsigned char> {
            cv::Mat frame = self.get_frame();
            
            if (frame.empty()) {
                return py::array_t<unsigned char>({});
            }
            
            if (!frame.isContinuous()) {
                frame = frame.clone();
            }
            
            auto output_frame = std::make_unique<cv::Mat>();
            if (frame.depth() != CV_8U) {
                double minVal, maxVal;
                cv::minMaxLoc(frame, &minVal, &maxVal);
                double scale = (maxVal > minVal) ? 255.0 / (maxVal - minVal) : 1.0;
                frame.convertTo(*output_frame, CV_8U, scale);
            } else {
                *output_frame = frame;
            }
            
            std::vector<size_t> shape;
            std::vector<size_t> strides;
            
            if (output_frame->channels() == 1) {
                shape = {
                    static_cast<size_t>(output_frame->rows), 
                    static_cast<size_t>(output_frame->cols)
                };
                strides = {
                    static_cast<size_t>(output_frame->step[0]), 
                    sizeof(unsigned char)
                };
            } else {
                shape = {
                    static_cast<size_t>(output_frame->rows), 
                    static_cast<size_t>(output_frame->cols), 
                    static_cast<size_t>(output_frame->channels())
                };
                strides = {
                    static_cast<size_t>(output_frame->step[0]), 
                    static_cast<size_t>(output_frame->step[1]), 
                    sizeof(unsigned char)
                };
            }
            
            auto* frame_ptr = output_frame.release(); 
            py::capsule capsule(frame_ptr, mat_deleter);
            
            return py::array_t<unsigned char>(
                shape, 
                strides, 
                frame_ptr->data, 
                capsule
            ); }, py::return_value_policy::take_ownership, "Gets the current frame from the camera as a NumPy array.\n"
                                                                                                                             "Returns an empty array if no frame is available.");

    m.attr("__version__") = "1.0.0";
}
