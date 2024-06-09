from pydantic import BaseModel
from typing import Union, List

class User(BaseModel):
    username: str
    password: str

class Camera(BaseModel):
    camera_id: int
    user_id: int
    name: str
    url: str
    points:str
    detected: bool
