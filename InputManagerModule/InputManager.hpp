#pragma once

#include <opencv2/opencv.hpp>

class InputManager {
  public:
    explicit InputManager(int camera_index = 0);
    ~InputManager();

    InputManager(const InputManager &) = delete;
    InputManager &operator=(const InputManager &) = delete;

    bool start_capture();
    void stop_capture();
    bool is_capture_open() const;

    cv::Mat get_frame();

    void set_normalization_params(float mean, float std);
    void set_mirror_enabled(bool enabled);

  private:
    cv::VideoCapture _capture;
    int _camera_index;
    bool _mirror_enabled;
    float _normalization_mean;
    float _normalization_std;

    bool reconnect_camera();
};
