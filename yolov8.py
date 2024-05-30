import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8m.pt")

# Open the video stream
# cap = cv2.VideoCapture("rtsp://admin:ECSIAQ@192.168.1.85:554")

# Open the video stream
cap = cv2.VideoCapture("rtsp://admin:JCNMJQ@192.168.1.22:554")

# Open the video stream
# cap = cv2.VideoCapture("rtsp://admin:PSJCJW@192.168.1.5:554")

# Get the current time
# start_time = datetime.now()

# Define the output video parameters
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_video = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

while True:
    _, frame = cap.read()
    if _:
        # Predict using YOLO model
        results = model.predict(frame)

        # Process the results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].tolist()
                c = box.cls
                if c == 0:  # Assuming class 0 represents person
                    x1, y1, x2, y2 = map(int, b)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write the frame to the output video
        # output_video.write(frame)

        # Show the frame
        cv2.imshow('YOLO V8 Detection', frame)

        # Check if 2 minutes have passed
        # elapsed_time = datetime.now() - start_time
        # if elapsed_time >= timedelta(minutes=2):
        #     break

        # Check for 'q' press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and writer objects
cap.release()
# output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
