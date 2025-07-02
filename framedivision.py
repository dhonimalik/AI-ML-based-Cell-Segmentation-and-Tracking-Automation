import cv2
import os
import shutil

if os.path.exists("output"):
    shutil.rmtree("output")
os.makedirs("output")

video_path = "raw.avi"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(total_frames / fps)

for sec in range(duration):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    success, frame = cap.read()
    if success:
        cv2.imwrite(f"output/frame_{sec}.jpg", frame)

cap.release()
print("✅ Done! All frames saved in 'output' folder.")
