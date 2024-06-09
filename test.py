from fastapi import FastAPI, Depends, Response, BackgroundTasks, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
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
from database import engine, get_db
from routes import camera
from sqlalchemy.orm import Session
import schemas
from queue import Queue

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

app.include_router(camera.router)

# Initialize YOLO model
model = YOLO("yolov8m.pt")

# Define a shared state dictionary to track the streaming state
streaming_state = {"active": True}

# Dictionary to keep track of camera threads and their stop events
# camera_threads = {}

# Dictionary to keep track of camera threads, queues, and stop events
camera_info = {}

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
def save_video(queue, db: Session, stop_event, detected: bool = False):
    try:
        if not detected:
            return
        # cap = cv2.VideoCapture(url)
        frame_rate_limit = 1000
        prev_frame_time = 0
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = None
        recording = False
        # while cap.isOpened():
        #     ret, frame = cap.read()
        while True:
            frame, url = queue.get()
            if(stop_event.is_set()):
                break
            if frame is None:
                print("Failed to capture frame")
                break

            current_frame_time = time.time()
            time_diff = current_frame_time - prev_frame_time
            if time_diff < 1.0 / frame_rate_limit:
                continue

            prev_frame_time = current_frame_time
            results = model.predict(frame, conf=0.42)[0]

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
                out = cv2.VideoWriter(video_path, fourcc, 5, (frame.shape[1], frame.shape[0]))
                recording = True
                print(f"Started recording: {video_path}")
                # Call the create_notification function here
                camera = db.query(models.Camera).filter(models.Camera.url == url).first()
                notification = Notification(
                    title="Person Detected",
                    description=f"A person was detected at {timestamp}",
                    videoUrl=f"http://0.0.0.0:8001/{video_path.replace('.avi', '.mp4')}",
                    camera_id=camera.camera_id,  # Example camera ID
                    user_id=camera.user_id     # Example user ID
                )
                asyncio.run(create_notification(notification))

            if recording:
                out.write(frame)

            if not person_detected and recording:
                out.release()
                recording = False
                print(f"Stopped recording")
                mp4_path = convert_avi_to_mp4(video_path)

        # cap.release()
        if recording:
            out.release()
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)

def capture_frames(url, queue: Queue, stop_event):
    try:
        print("#"*100)
        # Open video source (change the path to your video file or camera index)
        # Cam ngoai cua
        # cap = ffmpegcv.VideoCaptureStream(url)
        cap = cv2.VideoCapture(url)
        # cap = ffmpegcv.VideoCaptureStream("rtsp://admin:Cntt123a@namthan.ddns.net:554")
        # cap = cv2.VideoCapture("rtsp://admin:JCNMJQ@namthan.ddns.net:554", cv2.CAP_FFMPEG)
        # Cam su dung
        # cap = cv2.VideoCapture("rtsp://admin:PSJCJW@192.168.1.107:554")
        # Cam cong ty
        # cap = cv2.VideoCapture("rtsp://admin:ECSIAQ@192.168.1.85:554")
        # Set frame rate limit (e.g., 10 frames per second)
        frame_rate_limit = 10
        prev_frame_time = 0
        while cap.isOpened():
            if stop_event.is_set():
                break
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

            queue.put((frame, url))

        cap.release()
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)  # Adjust the retry interval as needed

# # Define a generator function to read frames from the video source
# def generate_frames(url):
#     try:
#         print("#"*100)
#         # Open video source (change the path to your video file or camera index)
#         # Cam ngoai cua
#         # cap = ffmpegcv.VideoCaptureStream(url)
#         cap = cv2.VideoCapture(url)
#         # cap = ffmpegcv.VideoCaptureStream("rtsp://admin:Cntt123a@namthan.ddns.net:554")
#         # cap = cv2.VideoCapture("rtsp://admin:JCNMJQ@namthan.ddns.net:554", cv2.CAP_FFMPEG)
#         # Cam su dung
#         # cap = cv2.VideoCapture("rtsp://admin:PSJCJW@192.168.1.107:554")
#         # Cam cong ty
#         # cap = cv2.VideoCapture("rtsp://admin:ECSIAQ@192.168.1.85:554")
#         # Set frame rate limit (e.g., 10 frames per second)
#         frame_rate_limit = 10
#         prev_frame_time = 0
#         while cap.isOpened():
#             if not streaming_state["active"]:
#                 break
#             if not cap.read():
#                 continue
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             if frame is None:
#                 print("Failed to capture frame")
#                 continue

#             # Get the current time
#             current_frame_time = time.time()
#             # Calculate the time difference
#             time_diff = current_frame_time - prev_frame_time

#             # If the time difference is less than the frame interval, skip this frame
#             if time_diff < 1.0 / frame_rate_limit:
#                 continue

#             # Update the previous frame time
#             prev_frame_time = current_frame_time

#             # Encode the frame as JPEG
#             # Resize frame for faster processing
#             # cropped_frame = cv2.resize(cropped_frame, (1280, 640))
#             # Predict using YOLO model
#             results = model.predict(frame)[0]
#             # Process the results
#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = box.cls
#                 if label == 0:  # Assuming class 0 represents person
#                     frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#             # img = Image.fromarray(cropped_frame)
#             # img_buffer = io.BytesIO()
#             # img.save(img_buffer, format='JPEG')
#             # frame_bytes = img_buffer.getvalue()

#             # yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         # Release video source when done
#         cap.release()
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         # Wait for some time before retrying
#         time.sleep(5)  # Adjust the retry interval as needed

# Define a generator function to read frames from the queue
def generate_frames(queue: Queue, detected: bool = False):
    while True:
        frame, url = queue.get()
        if frame is None:
            break
        if detected:
            results = model.predict(frame, conf=0.42)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = box.cls
                if label == 0:  # Assuming class 0 represents person
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
async def video_feed(url = "/", db: Session = Depends(get_db)):
    streaming_state["active"] = True
    global camera_info
    camera = db.query(models.Camera).filter(models.Camera.url == url).first()
    # if not camera:
    #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Camera with URL {url} not found')
    if url not in camera_info:
        camera_info[url] = {
            "queue": Queue(),
            "stop_event": threading.Event()
        }
        thread = camera_info[url]["thread"] = threading.Thread(target=capture_frames, args=(url, camera_info[url]["queue"], camera_info[url]["stop_event"]), daemon=True)
        thread.start()
        camera_info[url]["camera_thread"] = (thread, camera_info[url]["stop_event"])
    return StreamingResponse(generate_frames(camera_info[url]["queue"], camera.detected), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/camera", status_code=status.HTTP_201_CREATED)
def create(request: schemas.Camera, db: Session = Depends(get_db)):
    try:
        new_camera =models.Camera(camera_id=request.camera_id, user_id=request.user_id, name=request.name, url=request.url, points=request.points, detected=request.detected)
        db.add(new_camera)
        db.commit()
        db.refresh(new_camera)
        if request.url not in camera_info:
            camera_info[request.url] = {
                "queue": Queue(),
                "stop_event": threading.Event()
            }
            camera_info[request.url]["thread"] = threading.Thread(target=capture_frames, args=(request.url, camera_info[request.url]["queue"], camera_info[request.url]["stop_event"]), daemon=True)
            camera_info[request.url]["thread"].start()

        thread = threading.Thread(target=save_video, args=(camera_info[request.url]["queue"], db, camera_info[request.url]["stop_event"], new_camera.detected), daemon=True)
        thread.start()

        camera_info[request.url]["camera_thread"] = (thread, camera_info[request.url]["stop_event"])

        return new_camera
    except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(2)

@app.delete("/camera/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    try:
        camera = db.query(models.Camera).filter(models.Camera.camera_id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'camera with id {id} is not available')
        url = camera.url
        if url in camera_info:
            # Stop the thread and event associated with the camera's URL
            print(f"Stopping camera thread for URL: {url}" * 10)
            camera_thread, stop_event = camera_info[url]["camera_thread"]
            stop_event.set()
            camera_thread.join()

            # Remove camera information from the dictionary
            del camera_info[url]

        # Delete the camera from the database
        db.delete(camera)
        db.commit()

        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(2)
# @app.get("/videos/{filename}")
# async def get_video(filename: str):
#     file_path = os.path.join(video_folder, filename)
#     if os.path.exists(file_path):
#         return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")
#     else:
#         return {"error": "File not found"}

@app.get("/videos/{filename}")
async def get_video(filename: str):
    file_path = os.path.join(video_folder, filename)
    if os.path.exists(file_path):
        return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")
    return {"error": "File not found"}

@app.get("/stop-streaming")
async def stop_streaming():
    # Set the streaming state to inactive
    streaming_state["active"] = False
    return {"message": "Streaming stopped"}

# app.mount("/videos", StaticFiles(directory="videos"), name="videos")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
