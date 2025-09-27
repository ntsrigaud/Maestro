#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

class InputManager {
  public:
    InputManager(int camera_index = 0);
    ~InputManager();

    bool start_capture();
    void stop_capture();
    bool is_capture_open();
    cv::Mat get_frame();

  private:
    int _camera_index;
    cv::VideoCapture _capture;
};
