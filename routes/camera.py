from fastapi import APIRouter, Depends, status, HTTPException
import schemas, database, models
from sqlalchemy.orm import Session
from repository import camera
from fastapi.responses import StreamingResponse
import threading
import os
import signal

router = APIRouter(
  prefix="/camera",
  tags=["Camera"]
)
get_db = database.get_db
streams = []

@router.get('/{id}', status_code=200)
def show(id: int, db: Session = Depends(get_db)):
  cameraQuery =db.query(models.Camera).filter(models.Camera.camera_id == id).first()
  if not cameraQuery:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'camera with id {id} is not available')
  stop_event = threading.Event()
  t1 = threading.Thread(target=camera.show, args=(id, cameraQuery.url, stop_event), daemon=True)
  streams.append({"thread": t1, "stop_event": stop_event, "id": id})
  t1.start()
  return StreamingResponse(camera.show(id, cameraQuery.url, stop_event), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get('/delete/{id}', status_code=200)
def delete_stream(id: int):
    for stream in streams:
        if id == stream["id"]:
            stop_event = stream["stop_event"]
            stop_event.set()
            stream["thread"].join()  # Wait for the thread to finish
            streams.remove(stream)
            return {"detail": "Stream stopped"}
    raise HTTPException(status_code=404, detail="Stream not found")

# @router.post('', status_code=status.HTTP_201_CREATED)
# def create(request: schemas.Camera, db: Session = Depends(get_db)):
#   return camera.create(request, db)

@router.put('/{id}', status_code=status.HTTP_202_ACCEPTED)
def update(id, request: schemas.Camera, db: Session = Depends(get_db)):
  return camera.update(id, request, db)

# @router.delete('/{camera_id}', status_code=status.HTTP_204_NO_CONTENT)
# def destroy(camera_id, db: Session = Depends(get_db)):
#   return camera.destroy(camera_id, db)
