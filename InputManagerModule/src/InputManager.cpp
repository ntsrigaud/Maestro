#include "InputManager.hpp"
#include <chrono>
#include <mutex>
#include <thread>

InputManager::InputManager(int camera_index)
    : _camera_index(camera_index),
      _mirror_enabled(true),
      _normalization_mean(0.5f),
      _normalization_std(0.5f),
      _is_capturing(false) {}

InputManager::~InputManager() {
    stop_continuous_capture();
}

bool InputManager::is_capture_open() const {
    return _capture.isOpened();
}

bool InputManager::start_capture() {
    if (_capture.isOpened()) {
        return true;
    }

    _capture.open(_camera_index);
    if (!_capture.isOpened()) {
        return false;
    }

    _capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    _capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    return true;
}

void InputManager::stop_capture() {
    if (_capture.isOpened()) {
        _capture.release();
    }
}

bool InputManager::start_continuous_capture() {
    if (!start_capture()) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(_frame_mutex);
        _captured_frames.clear();
    }

    _is_capturing = true;
    _capture_thread = std::thread(&InputManager::capture_loop, this);
    return true;
}

void InputManager::stop_continuous_capture() {
    _is_capturing = false;
    if (_capture_thread.joinable()) {
        _capture_thread.join();
    }
    stop_capture();
}

void InputManager::capture_loop() {
    while (_is_capturing) {
        cv::Mat frame = get_frame();
        if (!frame.empty()) {
            FrameData frame_data;
            frame_data.frame = frame.clone();
            frame_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

            {
                std::lock_guard<std::mutex> lock(_frame_mutex);
                _latest_frame = frame_data;
                _captured_frames.push_back(frame_data);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

cv::Mat InputManager::get_frame() {
    if (!_capture.isOpened()) {
        return cv::Mat();
    }

    cv::Mat frame;
    if (!_capture.read(frame) || frame.empty()) {
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

FrameData InputManager::get_latest_frame() {
    std::lock_guard<std::mutex> lock(_frame_mutex);
    return _latest_frame;
}

std::vector<FrameData> InputManager::get_all_frames() {
    std::lock_guard<std::mutex> lock(_frame_mutex);
    return _captured_frames;
}

void InputManager::set_normalization_params(float mean, float std) {
    _normalization_mean = mean;
    _normalization_std = std;
}

void InputManager::set_mirror_enabled(bool enabled) {
    _mirror_enabled = enabled;
}
