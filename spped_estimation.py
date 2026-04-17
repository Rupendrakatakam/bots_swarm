from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2

model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("bus.mp4")
assert cap.isOpened(), "Error opening video file"
w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mjpg"), fps, (w,h))


line_pts = [(0, 500), (1000, 500)]

speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts, names=names,view_img=True,)
    
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error reading frame")
        break
    track = model.track(frame, persist=True, show = False)
    frame = speed_obj.estimate_speed(frame, track)
    video_writer.write(frame)
    
    

cap.release()
video_writer.release()
cv2.destroyAllWindows()