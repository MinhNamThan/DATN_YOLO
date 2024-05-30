from fastapi import APIRouter, Depends, status, HTTPException
import schemas, database, models
from sqlalchemy.orm import Session
from repository import camera

router = APIRouter(
  prefix="/camera",
  tags=["Camera"]
)
get_db = database.get_db

@router.get('/{id}', status_code=200)
def show(id, db: Session = Depends(get_db)):
  return camera.show(id, db)

@router.post('/', status_code=status.HTTP_201_CREATED)
def create(request: schemas.Camera, db: Session = Depends(get_db)):
  return camera.create(request, db)

@router.put('/{id}', status_code=status.HTTP_202_ACCEPTED)
def update(id, request: schemas.Camera, db: Session = Depends(get_db)):
  return camera.update(id, request, db)

@router.delete('/{id}', status_code=status.HTTP_204_NO_CONTENT)
def destroy(id, db: Session = Depends(get_db)):
  return camera.destroy(id, db)
