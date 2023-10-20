import cv2
from ultralytics import YOLO
import numpy as np

# derek was here
capture_object = cv2.VideoCapture("dogs.mp4")

# load detection model
model = YOLO("yolov8m.pt")

while True:
    returnValue, frame = capture_object.read()
    if returnValue == False:
        break

    ##ZOOM FRAME
    zoom_factor = 1.25

    # Get the dimensions of the frame
    height, width, channels = frame.shape

    # Calculate the new dimensions after zooming
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)

    # Resize the frame to the new dimensions
    zoomed_frame = cv2.resize(frame, (new_width, new_height))

    ## pass frame into model
    results = model(frame, device="mps")  # fast with gpu
    # results = model(frame)  # slow
    bounding_boxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
    print("bounding boxes:", bounding_boxes)

    classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
    for box, cls in zip(bounding_boxes, classes):
        (x1, y1, x2, y2) = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 0, 0), 2)
        cv2.putText(
            frame,
            model.names.get(cls.item()),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_PLAIN,
            4,
            (225, 0, 0),
            2,
        )

    cv2.imshow("Img", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # for nonblocking behavior
        break

# when finished release object, like file.close()

capture_object.release()
cv2.destroyAllWindows()
