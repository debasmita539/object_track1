import cv2
from ultralytics import YOLO

model = YOLO('yolov9e.pt')

video_path = r"Test.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = 640
frame_height = 480
while cap.isOpened():
   
    success, frame = cap.read()

    if success:
       
        conf = 0.2
        iou = 0.5
        results = model.track(frame, persist=True, conf=conf, iou=iou, save=True, tracker="bytetrack.yaml")
        annotated_frame = results[0].plot()

      
        resized_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
        cv2.imshow("YOLOv9 Tracking", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

