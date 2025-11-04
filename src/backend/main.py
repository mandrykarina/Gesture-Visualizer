import os
import io
from urllib.parse import quote_plus
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, select
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

# MinIO
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False
)

bucket_name = os.getenv("MINIO_BUCKET", "gesture-files")
try:
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    print("[OK] MinIO connected")
except Exception as e:
    print(f"[WARN] MinIO: {e}")

# PostgreSQL
user = os.getenv("POSTGRES_USER", "testuser")
password = os.getenv("POSTGRES_PASSWORD", "testpass123")
host = os.getenv("POSTGRES_HOST", "localhost")
port = os.getenv("POSTGRES_PORT", "5432")
db = os.getenv("POSTGRES_DB", "testdb")

DATABASE_URL = f"postgresql+asyncpg://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"

engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_size=5,
    max_overflow=10
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

class FileRecord(Base):
    __tablename__ = "files"
    id: int = Column(Integer, primary_key=True)
    filename: str = Column(String, nullable=False)
    s3_path: str = Column(String, nullable=False)

# Инициализация таблиц - БЕЗ автоматического создания
async def init_db():
    """Должна быть вызвана один раз вручную если таблицы не созданы"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("[OK] Database initialized")
    except Exception as e:
        print(f"[WARN] Database init: {e}")

@app.get("/")
def root():
    return {"message": "API работает"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        s3_key = file.filename

        minio_client.put_object(
            bucket_name,
            s3_key,
            io.BytesIO(content),
            length=len(content),
            content_type=file.content_type or "application/octet-stream"
        )

        async with AsyncSessionLocal() as session:
            s3_path = f"s3://{bucket_name}/{s3_key}"
            db_file = FileRecord(filename=file.filename, s3_path=s3_path)
            session.add(db_file)
            await session.commit()

        return {"status": "ok", "file": file.filename, "path": s3_path}
    except Exception as e:
        print(f"[ERROR] Upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(FileRecord))
            files = result.scalars().all()
            return [
                {"id": f.id, "filename": f.filename, "s3_path": f.s3_path}
                for f in files
            ]
    except Exception as e:
        print(f"[ERROR] List files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/init-db")
async def init_database():
    """Вручную создай таблицы при первом запуске"""
    await init_db()
    return {"status": "ok", "message": "Database initialized"}
