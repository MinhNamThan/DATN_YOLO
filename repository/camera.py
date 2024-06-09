from sqlalchemy.orm import Session
import models, schemas
from fastapi import HTTPException, status
import cv2
import time
from ultralytics import YOLO
import torch
import threading
from pydantic import BaseModel
import httpx
import asyncio
import ffmpeg
import os
from datetime import datetime

# Dictionary to keep track of camera threads and their stop events
camera_threads = {}
# Initialize YOLO model
model = YOLO("yolov8m-seg.pt")

# Define a shared state dictionary to track the streaming state
streaming_state = {"active": True}

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
def save_video(url, db: Session, stop_event):
    try:
        print('save_video'*10)
        cap = cv2.VideoCapture(url)
        frame_rate_limit = 10
        prev_frame_time = 0
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = None
        recording = False
        while cap.isOpened():
            ret, frame = cap.read()
            if stop_event.is_set():
                break
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
                out = cv2.VideoWriter(video_path, fourcc, 4, (frame.shape[1], frame.shape[0]))
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
                # threading.Thread(target=convert_avi_to_mp4, args=(video_path,), daemon=True).start()

        cap.release()
        if recording:
            out.release()
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)

def create(request: schemas.Camera, db: Session):
  try:
    new_camera =models.Camera(camera_id=request.camera_id, user_id=request.user_id, name=request.name, url=request.url, points=request.points)
    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)
    # db = next(db)
    url = request.url  # Replace with your video stream URL
    stop_event = threading.Event()
    thread = threading.Thread(target=save_video, args=(url,db, stop_event), daemon=True)
    thread.start()

    # Store the thread and stop event
    camera_threads[new_camera.camera_id] = (thread, stop_event)
    return new_camera
  except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(2)

def destroy(camera_id: int, db: Session):
  try:
    camera = db.query(models.Camera).filter(models.Camera.camera_id == camera_id)
    if not camera.first():
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'camera with id {id} is not available')

    # Stop the thread if it's running
    if camera_id in camera_threads:
        thread, stop_event = camera_threads[camera_id]
        stop_event.set()
        thread.join()
        del camera_threads[camera_id]
    camera.delete(synchronize_session=False)
    db.commit()
    return 'done'
  except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(2)

def update(id: int, request: schemas.Camera, db:Session):
  camera = db.query(models.Camera).filter(models.Camera.id == id)
  if not camera.first():
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'camera with id {id} is not available')
  camera.update(request.dict())
  db.commit()
  return 'updated'

def show(id: int, url: str, stop_event):
  print("#"*100)
  try:
    # Open video source (change the path to your video file or camera index)
    # Cam ngoai cua
    # cap = ffmpegcv.VideoCaptureStream(url)
    print(torch.cuda.is_available())
    cap = cv2.VideoCapture(url)
    # cap = ffmpegcv.VideoCaptureStream("rtsp://admin:JCNMJQ@namthan.ddns.net:554")
    # cap = cv2.VideoCapture("rtsp://admin:JCNMJQ@namthan.ddns.net:554", cv2.CAP_FFMPEG)
    # Cam su dung
    # cap = cv2.VideoCapture("rtsp://admin:PSJCJW@192.168.1.107:554")
    # Cam cong ty
    # cap = cv2.VideoCapture("rtsp://admin:ECSIAQ@192.168.1.85:554")
    # Set frame rate limit (e.g., 10 frames per second)
    # frame_rate_limit = 100
    # prev_frame_time = 0
    while cap.isOpened():
        if stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            continue
        if frame is None:
            print("Failed to capture frame")
            continue

        # # Get the current time
        # current_frame_time = time.time()
        # # Calculate the time difference
        # time_diff = current_frame_time - prev_frame_time

        # # If the time difference is less than the frame interval, skip this frame
        # if time_diff < 1.0 / frame_rate_limit:
        #     continue

        # # Update the previous frame time
        # prev_frame_time = current_frame_time

        # Encode the frame as JPEG
        # Resize frame for faster processing
        # cropped_frame = cv2.resize(cropped_frame, (1280, 640))
        # Predict using YOLO model
        # Initialize YOLO model
        model = YOLO("yolov8n.pt")
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
