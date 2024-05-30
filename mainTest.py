from fastapi import FastAPI, Response, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
import cv2
from ultralytics import YOLO
import time


app = FastAPI()

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

# Define a generator function to read frames from the video source
async def generate_frames():
    try:
        # Initialize YOLO model
        model = YOLO("yolov8s.pt")
        # Open video source (change the path to your video file or camera index)
        # Cam ngoai cua
        cap = cv2.VideoCapture("rtsp://admin:JCNMJQ@namthan.ddns.net:554", cv2.CAP_FFMPEG)
        # Cam su dung
        # cap = cv2.VideoCapture("rtsp://admin:PSJCJW@192.168.1.107:554")
        # Cam cong ty
        # cap = cv2.VideoCapture("rtsp://admin:ECSIAQ@192.168.1.85:554")
        w1, h1, w2, h2 = 150, 50, 1800, 1800
        while cap.isOpened():
            ret, frame = cap.read()
            # if not ret:
            #     break
            cropped_frame = frame
            # Encode the frame as JPEG
            # Resize frame for faster processing
            # cropped_frame = cv2.resize(cropped_frame, (1280, 640))
            # Predict using YOLO model
            results = model.predict(cropped_frame)

            # Process the results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].tolist()
                    c = box.cls
                    if c == 0:  # Assuming class 0 represents person
                        x1, y1, x2, y2 = map(int, b)
                        cropped_frame = cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', cropped_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Release video source when done
        cap.release()
    except Exception as e:
        print(f"Error occurred: {e}")
        # Wait for some time before retrying
        time.sleep(5)  # Adjust the retry interval as needed

@app.get("/")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
