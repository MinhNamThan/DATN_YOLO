from fastapi import FastAPI, Depends, Response, Request, HTTPException, status
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
from sqlalchemy.orm import Session
import schemas
from queue import Queue
import ast
import numpy as np

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

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(CustomCORSMiddleware)

models.Base.metadata.create_all(engine)

model = YOLO("yolov8m.pt")

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
            response = await client.post('https://leech-just-multiply.ngrok-free.app/notifications', json=notification.dict())
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses
            data = response.json()  # Parse the JSON response
            return data
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"Error calling external API: {exc.response.text}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")

def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path.replace('.avi', '.mp4')
    ffmpeg.input(avi_path).output(mp4_path).run()
    os.remove(avi_path)
    return mp4_path

def save_video(queue, db: Session, stop_event, detected: bool = False):
    try:
        if not detected:
            return
        frame_rate_limit = 10
        prev_frame_time = 0
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = None
        recording = False
        while True:
            frame, crop_frame, fix_size, url = queue.get()
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
            results = model.predict(crop_frame, conf=0.42, device='cpu', classes=0)[0]

            person_detected = False
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = box.cls
                if label == 0:
                    frame = cv2.rectangle(frame, (x1 + fix_size[1], y1 + fix_size[0]), (x2 + fix_size[1], y2 + fix_size[0]), (0, 255, 0), 2)
                    person_detected = True

            if person_detected and not recording:
                timestamp = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
                timestampNameFile = datetime.now().strftime("%Y%m%d%_H%M%S")
                video_path = os.path.join(video_folder, f"person_detected_{timestampNameFile}.avi")
                out = cv2.VideoWriter(video_path, fourcc, 3, (frame.shape[1], frame.shape[0]))
                recording = True
                print(f"Started recording: {video_path}")

                camera = db.query(models.Camera).filter(models.Camera.url == url).first()
                notification = Notification(
                    title="Phát hiện ngừoi",
                    description=f"Phát hiện người vào lúc {timestamp}",
                    videoUrl=f"https://improved-nicely-imp.ngrok-free.app/{video_path.replace('.avi', '.mp4')}",
                    camera_id=camera.camera_id,
                    user_id=camera.user_id
                )
                asyncio.run(create_notification(notification))

            if recording:
                out.write(frame)

            if not person_detected and recording:
                out.release()
                recording = False
                print(f"Stopped recording")
                convert_avi_to_mp4(video_path)

        if recording:
            out.release()
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)

def capture_frames(url, queue: Queue, stop_event, points):
    try:
        arrayPoints = ast.literal_eval(points)

        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cap = ffmpegcv.VideoCaptureStream("rtsp://admin:Cntt123a@namthan.ddns.net:554")
        frame_rate_limit = 10
        prev_frame_time = 0

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        points_pixel = []
        if len(arrayPoints) == 4:
            if(arrayPoints[0] > arrayPoints[2]):
                arrayPoints[0], arrayPoints[2] = arrayPoints[2], arrayPoints[0]
            if(arrayPoints[1] > arrayPoints[3]):
                arrayPoints[1], arrayPoints[3] = arrayPoints[3], arrayPoints[1]
            points_pixel = [[int(arrayPoints[0] * frame_width), int(arrayPoints[1] * frame_height)], [int(arrayPoints[2] * frame_width), int(arrayPoints[1] * frame_height)], [int(arrayPoints[2] * frame_width), int(arrayPoints[3] * frame_height)], [int(arrayPoints[0] * frame_width), int(arrayPoints[3] * frame_height)]]
        elif len(arrayPoints) > 4 & len(arrayPoints) % 2 == 0:
            for i in range(0, len(arrayPoints), 2):
                points_pixel.append([int(arrayPoints[i] * frame_width), int(arrayPoints[i + 1] * frame_height)])

        if not cap.isOpened():
            del camera_info[url]
            raise ValueError("Unable to open video stream")
        while cap.isOpened():
            if stop_event.is_set():
                print("Stopping thread")
                break
            if not cap.read():
                print("Failed to read frame")
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame")
                break

            current_frame_time = time.time()
            time_diff = current_frame_time - prev_frame_time

            if time_diff < 1.0 / frame_rate_limit:
                continue

            prev_frame_time = current_frame_time
            if len(points_pixel) > 0:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                points_array = np.array([points_pixel], dtype=np.int32)
                cv2.fillPoly(mask, points_array, 255)

                crop_frame = cv2.bitwise_and(frame, frame, mask=mask)

                x, y, w, h = cv2.boundingRect(points_array)
                crop_frame = crop_frame[y:y+h, x:x+w]
                fix_size = [y, x]
            else:
                crop_frame = frame
                fix_size = [0, 0]
            if queue.qsize() < 5:
                queue.put((frame, crop_frame, fix_size, url))
        cap.release()
        return
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)

def generate_frames(queue: Queue, streaming_active: threading.Event, detected: bool = False):
    try:
        while streaming_active.is_set():
            frame, crop_frame, fix_size, url = queue.get()
            if crop_frame is None:
                print("Failed to capture frame")
                continue
            if detected:
                results = model.predict(crop_frame, conf=0.42, save=False, save_txt=False, show=False, classes=0)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = box.cls
                    if label == 0:
                        frame = cv2.rectangle(frame, (x1 + fix_size[1], y1 + fix_size[0]), (x2 + fix_size[1], y2 + fix_size[0]), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            data_encode = np.array(buffer)
            frame_bytes = data_encode.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(5)

@app.get("/")
async def video_feed(url = "/", username = "", password = "", db: Session = Depends(get_db)):
    box = db.query(models.User).filter(models.User.username == username, models.User.password == password).first()
    if not box:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'User with username {username} not found')
    global camera_info
    camera = db.query(models.Camera).filter(models.Camera.url == url).first()
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Camera with URL {url} not found')
    if url not in camera_info:
        print("Creating new thread")
        camera_info[url] = {
            "queue": Queue(maxsize=10),
            "stop_event": threading.Event(),
            "streaming_active": threading.Event()
        }
        thread = camera_info[url]["thread"] = threading.Thread(target=capture_frames, args=(url, camera_info[url]["queue"], camera_info[url]["stop_event"], camera.points), daemon=True)
        thread.start()
        camera_info[url]["camera_thread"] = (thread, camera_info[url]["stop_event"])
    else:
        if not camera_info[url]["camera_thread"][0].is_alive():
            print("Thread is not alive")
            camera_info[url]["stop_event"].clear()
            camera_info[url]["thread"] = threading.Thread(target=capture_frames, args=(url, camera_info[url]["queue"], camera_info[url]["stop_event"], camera.points), daemon=True)
            camera_info[url]["thread"].start()
            camera_info[url]["camera_thread"] = (camera_info[url]["thread"], camera_info[url]["stop_event"])
    camera_info[url]["streaming_active"].set()
    return StreamingResponse(generate_frames(camera_info[url]["queue"], camera_info[url]["streaming_active"], camera.detected), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/camera", status_code=status.HTTP_201_CREATED)
def create(request: schemas.Camera, db: Session = Depends(get_db)):
    try:
        new_camera =models.Camera(camera_id=request.camera_id, user_id=request.user_id, name=request.name, url=request.url, points=request.points, detected=request.detected)
        db.add(new_camera)
        db.commit()
        db.refresh(new_camera)
        if request.url not in camera_info:
            print("Creating new thread")
            camera_info[request.url] = {
                "queue": Queue(maxsize=10),
                "stop_event": threading.Event(),
                "streaming_active": threading.Event()
            }
            camera_info[request.url]["thread"] = threading.Thread(target=capture_frames, args=(request.url, camera_info[request.url]["queue"], camera_info[request.url]["stop_event"], request.points), daemon=True)
            camera_info[request.url]["thread"].start()
            camera_info[request.url]["camera_thread"] = (camera_info[request.url]["thread"], camera_info[request.url]["stop_event"])

        thread = threading.Thread(target=save_video, args=(camera_info[request.url]["queue"], db, camera_info[request.url]["stop_event"], new_camera.detected), daemon=True)
        thread.start()


        return new_camera
    except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(2)

@app.put("/camera/{camera_id}", status_code=status.HTTP_202_ACCEPTED)
def update(request: schemas.Camera, camera_id: int, db: Session = Depends(get_db)):
    try:
        print(request.dict())
        camera_query = db.query(models.Camera).filter(models.Camera.camera_id == camera_id)
        camera = camera_query.first()
        if not camera:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera with id {camera_id} not found")

        old_url = camera.url
        new_url = request.url
        if old_url in camera_info:
            print("del"*10)
            camera_info[old_url]["streaming_active"].clear()
            camera_info[old_url]["stop_event"].set()
            camera_info[old_url]["thread"].join()
            del camera_info[old_url]

        if new_url not in camera_info:
            print("new"*10)
            camera_info[new_url] = {
                "queue": Queue(maxsize=10),
                "stop_event": threading.Event(),
                "streaming_active": threading.Event()
            }
            thread = threading.Thread(target=capture_frames, args=(new_url, camera_info[new_url]["queue"], camera_info[new_url]["stop_event"], request.points), daemon=True)
            camera_info[new_url]["thread"] = thread
            thread.start()
            camera_info[new_url]["camera_thread"] = (thread, camera_info[new_url]["stop_event"])

        thread = threading.Thread(target=save_video, args=(camera_info[request.url]["queue"], db, camera_info[request.url]["stop_event"], request.detected), daemon=True)
        thread.start()

        camera_query.update(request.dict())
        db.commit()
        return "Updated"
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"fail")

@app.delete("/camera/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    try:
        camera = db.query(models.Camera).filter(models.Camera.camera_id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'camera with id {id} is not available')
        url = camera.url
        if url in camera_info:
            try:
                camera_thread, stop_event = camera_info[url]["camera_thread"]
                camera_info[url]["streaming_active"].clear()
                stop_event.set()
                camera_thread.join()
            except Exception as e:
                print(f"Error occurred: {e}")

            del camera_info[url]

        db.delete(camera)
        db.commit()

        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(2)

@app.get("/videos/{filename}")
async def get_video(filename: str):
    file_path = os.path.join(video_folder, filename)
    if os.path.exists(file_path):
        return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")
    return {"error": "File not found"}

@app.get("/stop_stream")
async def stop_streaming(urllink = ""):
    global camera_info
    print(camera_info)
    for url in camera_info:
        if(urllink != url):
            print("Stopping thread"*10)
            print(url)
            print(urllink)
            camera_info[url]["streaming_active"].clear()

async def stop_streaming_by_url(url = ""):
    global camera_info
    for url in camera_info:
        if(url == url):
            camera_info[url]["streaming_active"].clear()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
