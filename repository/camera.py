from sqlalchemy.orm import Session
import models, schemas
from fastapi import HTTPException, status

def get_all(db: Session):
  users = db.query(models.User).all()
  return users

def create(request: schemas.User, db: Session):
  new_user =models.User(email=request.email, password=Hash.bcrypt(request.password))
  db.add(new_user)
  db.commit()
  db.refresh(new_user)
  return new_user

def destroy(id: int, db: Session):
  user = db.query(models.User).filter(models.User.id == id)
  if not user.first():
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'user with id {id} is not available')
  user.delete(synchronize_session=False)
  db.commit()
  return 'done'

def update(id: int, request: schemas.User, db:Session):
  user = db.query(models.User).filter(models.User.id == id)
  if not user.first():
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'user with id {id} is not available')
  user.update(request.dict())
  db.commit()
  return 'updated'

def show(id: int, db: Session):
  camera =db.query(models.Camera).filter(models.Camera.id == id).first()
  if not camera:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'user with id {id} is not available')
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
