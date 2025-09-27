#include "InputManager.hpp"
#include <chrono>
#include <iostream>
#include <thread>

InputManager::InputManager(int camera_index)
    : _camera_index(camera_index),
      _mirror_enabled(true),
      _normalization_mean(0.5f),
      _normalization_std(0.5f) {}

InputManager::~InputManager() {
    stop_capture();
}

bool InputManager::is_capture_open() const {
    return _capture.isOpened();
}

bool InputManager::start_capture() {
    if (_capture.isOpened()) {
        std::cerr << "WARNING: Camera already opened" << std::endl;
        return true;
    }

    _capture.open(_camera_index);
    if (!_capture.isOpened()) {
        std::cerr << "ERROR: Could not open camera at index " << _camera_index << std::endl;
        return false;
    }

    _capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    _capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    _capture.set(cv::CAP_PROP_FPS, 30);

    cv::Mat test_frame;
    if (!_capture.read(test_frame) || test_frame.empty()) {
        std::cerr << "ERROR: Camera opened but cannot read frames" << std::endl;
        stop_capture();
        return false;
    }

    std::cout << "Camera started successfully: "
              << test_frame.cols << "x" << test_frame.rows << std::endl;
    return true;
}

void InputManager::stop_capture() {
    if (_capture.isOpened()) {
        _capture.release();
        std::cout << "Camera stopped" << std::endl;
    }
}

bool InputManager::reconnect_camera() {
    std::cout << "Attempting to reconnect camera..." << std::endl;
    stop_capture();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    return start_capture();
}

cv::Mat InputManager::get_frame() {
    if (!_capture.isOpened()) {
        std::cerr << "ERROR: Camera not opened" << std::endl;
        return cv::Mat();
    }

    cv::Mat frame;
    int retry_count = 0;
    const int max_retries = 3;

    while (retry_count < max_retries) {
        if (_capture.read(frame)) {
            if (!frame.empty()) {
                break;
            }
        }

        retry_count++;
        std::cerr << "WARNING: Failed to read frame (attempt " << retry_count << ")" << std::endl;

        if (retry_count < max_retries) {
            if (!reconnect_camera()) {
                std::cerr << "ERROR: Reconnection failed" << std::endl;
                return cv::Mat();
            }
        }
    }

    if (frame.empty()) {
        std::cerr << "ERROR: All frame read attempts failed" << std::endl;
        return cv::Mat();
    }

    if (_mirror_enabled) {
        cv::flip(frame, frame, 1);
    }

    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

    cv::Mat resized_frame;
    cv::resize(rgb_frame, resized_frame, cv::Size(224, 224));

    cv::Mat float_frame;
    resized_frame.convertTo(float_frame, CV_32FC3, 1.0 / 255.0);

    cv::Mat normalized_frame;
    cv::Mat mean_mat(224, 224, CV_32FC3, cv::Scalar(_normalization_mean, _normalization_mean, _normalization_mean));
    cv::Mat std_mat(224, 224, CV_32FC3, cv::Scalar(_normalization_std, _normalization_std, _normalization_std));

    cv::subtract(float_frame, mean_mat, normalized_frame);
    cv::divide(normalized_frame, std_mat, normalized_frame);

    return normalized_frame;
}

void InputManager::set_normalization_params(float mean, float std) {
    _normalization_mean = mean;
    _normalization_std = std;
}

void InputManager::set_mirror_enabled(bool enabled) {
    _mirror_enabled = enabled;
}
