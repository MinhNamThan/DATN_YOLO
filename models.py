from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from database import Base
from sqlalchemy.orm import relationship

class User(Base):
  __tablename__ = "users"

  id = Column(Integer, primary_key=True, index=True)
  username = Column(String)
  password = Column(String)

class Camera(Base):
  __tablename__ = "cameras"

  id = Column(Integer, primary_key=True, index=True)
  camera_id = Column(String)
  user_id = Column(String)
  name = Column(String)
  url = Column(String)
  points = Column(String)
