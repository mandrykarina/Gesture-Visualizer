import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv(encoding='utf-8')

app = FastAPI(title="Gesture File Upload API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== MinIO ========
try:
    minio_client = Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False
    )
    
    bucket_name = os.getenv("MINIO_BUCKET", "gesture-files")
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    print("[OK] MinIO connected")
except Exception as e:
    print(f"[ERROR] MinIO failed: {e}")
    minio_client = None

# ======== SQLite ========
DATABASE_URL = "sqlite:///./gesture.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class FileRecord(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    s3_path = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)
print("[OK] SQLite database initialized")

# ======== API ========
@app.get("/")
def root():
    return {
        "message": "API работает!",
        "minio": "OK" if minio_client else "OFFLINE",
        "database": "SQLite OK"
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not minio_client:
            raise HTTPException(status_code=503, detail="MinIO offline")
        
        print(f"[INFO] Uploading: {file.filename}")
        
        # Читай файл
        content = await file.read()
        file_size = len(content)
        print(f"[INFO] File size: {file_size} bytes")
        
        # Загружай в MinIO
        s3_key = file.filename
        print(f"[INFO] Uploading to MinIO: {s3_key}")
        
        minio_client.put_object(
            bucket_name,
            s3_key,
            io.BytesIO(content),
            length=file_size,
            content_type=file.content_type or "application/octet-stream"
        )
        print(f"[OK] MinIO upload complete")

        # Сохрани в SQLite
        db = SessionLocal()
        try:
            s3_path = f"s3://{bucket_name}/{s3_key}"
            db_file = FileRecord(filename=file.filename, s3_path=s3_path)
            db.add(db_file)
            db.commit()
            db.refresh(db_file)
            print(f"[OK] Database saved: {db_file.id}")
        finally:
            db.close()

        return {
            "status": "ok",
            "file": file.filename,
            "size": file_size,
            "path": s3_path
        }
    
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.get("/files")
def list_files():
    try:
        db = SessionLocal()
        try:
            files = db.query(FileRecord).all()
            print(f"[INFO] Found {len(files)} files")
            return [
                {"id": f.id, "filename": f.filename, "s3_path": f.s3_path}
                for f in files
            ]
        finally:
            db.close()
    except Exception as e:
        print(f"[ERROR] List files failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    """Проверка здоровья API"""
    return {
        "status": "ok",
        "minio": "connected" if minio_client else "offline",
        "database": "ok"
    }
