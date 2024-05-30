from fastapi import FastAPI, Response, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ultralytics import YOLO
import time
import cv2
import os
from datetime import datetime
import threading
from pydantic import BaseModel
import httpx
import asyncio
import ffmpeg
import models
from database import engine

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

models.Base.metadata.create_all(engine)

# Initialize YOLO model
model = YOLO("yolov8s.pt")

video_folder = "videos"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

class Notification(BaseModel):
    title: str
    description: str
    videoUrl: str
    camera_id: int
    user_id: int

async def create_notification(notification: Notification):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post('http://localhost:8000/notifications', json=notification.dict())
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses
            data = response.json()  # Parse the JSON response
            return data
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"Error calling external API: {exc.response.text}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")

# Function to convert .avi to .mp4
def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path.replace('.avi', '.mp4')
    ffmpeg.input(avi_path).output(mp4_path).run()
    os.remove(avi_path)
    return mp4_path

# Define a function to save videos when a person is detected
def save_video(url):
    try:
        cap = cv2.VideoCapture(url)
        frame_rate_limit = 1000
        prev_frame_time = 0
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = None
        recording = False
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                print("Failed to capture frame")
                break

            current_frame_time = time.time()
            time_diff = current_frame_time - prev_frame_time
            if time_diff < 1.0 / frame_rate_limit:
                continue

            prev_frame_time = current_frame_time
            results = model.predict(frame)[0]

            person_detected = False
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = box.cls
                if label == 0:  # Assuming class 0 represents person
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    person_detected = True

            if person_detected and not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(video_folder, f"person_detected_{timestamp}.avi")
                out = cv2.VideoWriter(video_path, fourcc, 10, (frame.shape[1], frame.shape[0]))
                recording = True
                print(f"Started recording: {video_path}")
                # Call the create_notification function here
                notification = Notification(
                    title="Person Detected",
                    description=f"A person was detected at {timestamp}",
                    videoUrl=f"http://0.0.0.0:8001/{video_path}",
                    camera_id=1,  # Example camera ID
                    user_id=1     # Example user ID
                )
                asyncio.run(create_notification(notification))

            if recording:
                out.write(frame)

            if not person_detected and recording:
                out.release()
                recording = False
                print(f"Stopped recording")
                # mp4_path = convert_avi_to_mp4(video_path)

        cap.release()
        if recording:
            out.release()
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)

# Define a generator function to read frames from the video source
async def generate_frames(url):
    try:
        # Open video source (change the path to your video file or camera index)
        # Cam ngoai cua
        # cap = ffmpegcv.VideoCaptureStream(url)
        cap = cv2.VideoCapture(url)
        # cap = ffmpegcv.VideoCaptureStream("rtsp://admin:JCNMJQ@namthan.ddns.net:554")
        # cap = cv2.VideoCapture("rtsp://admin:JCNMJQ@namthan.ddns.net:554", cv2.CAP_FFMPEG)
        # Cam su dung
        # cap = cv2.VideoCapture("rtsp://admin:PSJCJW@192.168.1.107:554")
        # Cam cong ty
        # cap = cv2.VideoCapture("rtsp://admin:ECSIAQ@192.168.1.85:554")
        # Set frame rate limit (e.g., 10 frames per second)
        frame_rate_limit = 10
        prev_frame_time = 0
        while cap.isOpened():
            if not cap.read():
                continue
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
        time.sleep(5)  # Adjust the retry interval as needed

@app.get("/")
async def video_feed(url = "/"):
    return StreamingResponse(generate_frames(url), media_type="multipart/x-mixed-replace; boundary=frame")

def start_save_video_task():
    url = "rtsp://admin:PSJCJW@192.168.1.3:554"  # Replace with your video stream URL
    threading.Thread(target=save_video, args=(url,), daemon=True).start()

# @app.get("/videos/{filename}")
# async def get_video(filename: str):
#     file_path = os.path.join(video_folder, filename)
#     if os.path.exists(file_path):
#         return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")
#     else:
#         return {"error": "File not found"}

if __name__ == "__main__":
    # start_save_video_task()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
