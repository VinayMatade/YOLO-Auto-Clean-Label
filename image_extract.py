import cv2
import os


def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    print(f"Opening video: {os.path.abspath(video_path)}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(
            output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Done! Extracted {frame_count} frames to '{output_folder}'")


# Just edit these values
video_path = "/home/vinay/Projects/YOLO-Auto-Clean-Label/test16/test16.mp4"
output_folder = "frames5"

extract_frames(video_path, output_folder)
