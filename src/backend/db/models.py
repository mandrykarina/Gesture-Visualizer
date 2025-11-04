from sqlalchemy import Column, Integer, String, DateTime, func
from db.session import Base

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=True)
    s3_path = Column(String, nullable=False)
    status = Column(String, default="uploaded")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
