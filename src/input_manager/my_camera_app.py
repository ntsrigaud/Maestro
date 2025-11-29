from input_manager import InputManager
import time

if __name__ == "__main__":
    manager = InputManager(camera_index=0)
    manager.start_continuous_capture()

    for i in range(100):
        frame_data = manager.get_latest_frame()
        if frame_data:
            frame = frame_data.to_numpy()
            print(f"Frame {i}: {frame.shape}, timestamp= {frame_data.timestamp}")
            time.sleep(0.033)

    manager.stop_continuous_capture()
    print("Done!")
