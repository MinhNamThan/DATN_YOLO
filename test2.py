from fastapi import FastAPI, Response, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ultralytics import YOLO
import time
import ffmpegcv
from PIL import Image
import io
import cv2
import torch

app = FastAPI(openapi_url="/openapi.json")

class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Length,Content-Range'
        return response

app = FastAPI(
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Set up CORS middleware for API routes
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply the custom CORS middleware
app.add_middleware(CustomCORSMiddleware)

# Initialize YOLO model
model = YOLO("yolov8s.pt")

# Define a generator function to read frames from the video source
async def generate_frames(url):

    try:
        # Open video source (change the path to your video file or camera index)
        # Cam ngoai cua
        # cap = ffmpegcv.VideoCaptureStream(url)
        print(torch.cuda.is_available())
        cap = ffmpegcv.VideoCaptureStream("rtsp://admin:JCNMJQ@namthan.ddns.net:554")
        # cap = cv2.VideoCapture("rtsp://admin:JCNMJQ@namthan.ddns.net:554", cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY ])
        # Cam su dung
        # cap = cv2.VideoCapture("rtsp://admin:PSJCJW@192.168.1.107:554")
        # Cam cong ty
        # cap = cv2.VideoCapture("rtsp://admin:ECSIAQ@192.168.1.85:554")
        # Set frame rate limit (e.g., 10 frames per second)
        frame_rate_limit = 10
        prev_frame_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            if frame is None:
                print("Failed to capture frame")
                continue

            # Get the current time
            current_frame_time = time.time()
            # Calculate the time difference
            time_diff = current_frame_time - prev_frame_time

            # If the time difference is less than the frame interval, skip this frame
            if time_diff < 1.0 / frame_rate_limit:
                continue

            # Update the previous frame time
            prev_frame_time = current_frame_time
            frame
            # Encode the frame as JPEG
            # Resize frame for faster processing
            # cropped_frame = cv2.resize(cropped_frame, (1280, 640))
            # Predict using YOLO model
            results = model.predict(frame)[0]

            # Process the results
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = box.cls
                if label == 0:  # Assuming class 0 represents person
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # img = Image.fromarray(cropped_frame)
            # img_buffer = io.BytesIO()
            # img.save(img_buffer, format='JPEG')
            # frame_bytes = img_buffer.getvalue()

            # yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Release video source when done
        cap.release()
    except Exception as e:
        print(f"Error occurred: {e}")
        # Wait for some time before retrying
        time.sleep(5)  # Adjust the retry interval as neede
@app.get("/")
async def video_feed(url = "/"):
    return StreamingResponse(generate_frames(url), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
