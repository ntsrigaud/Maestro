#pragma once
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

struct FrameData {
    cv::Mat frame;
    int64_t timestamp;
    FrameData() : timestamp(0) {}
};

class InputManager {
  public:
    InputManager(int camera_index = 0);
    ~InputManager();

    bool is_capture_open() const;
    bool start_capture();
    void stop_capture();

    bool start_continuous_capture();
    void stop_continuous_capture();
    FrameData get_latest_frame();
    std::vector<FrameData> get_all_frames();

    void set_normalization_params(float mean, float std);
    void set_mirror_enabled(bool enabled);

  private:
    void capture_loop();
    cv::Mat get_frame();

    int _camera_index;
    bool _mirror_enabled;
    float _normalization_mean;
    float _normalization_std;
    bool _is_capturing;

    cv::VideoCapture _capture;
    std::thread _capture_thread;
    std::mutex _frame_mutex;
    FrameData _latest_frame;
    std::vector<FrameData> _captured_frames;
};
