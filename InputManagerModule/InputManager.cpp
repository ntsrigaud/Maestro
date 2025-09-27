#include "InputManager.hpp"
#include <iostream>

InputManager::InputManager(int camera_index) : _camera_index(camera_index) {}

InputManager::~InputManager() { stop_capture(); }

bool InputManager::is_capture_open() { return _capture.isOpened(); }

bool InputManager::start_capture() {
    _capture.open(_camera_index);
    if (!_capture.isOpened()) {
        std::cerr << "ERROR: Could not open camera at index" << _camera_index << std::endl;
        return false;
    }
    return true;
}

void InputManager::stop_capture() {
    if (_capture.isOpened()) {
        _capture.release();
    }
}

cv::Mat InputManager::get_frame() {
    cv::Mat frame;
    if (_capture.read(frame)) {
        return frame;
    }
    return cv::Mat();
}
