import os
from fastapi import APIRouter, UploadFile, HTTPException
from s3.client import minio_client
from config import settings
from db.session import SessionLocal
from db.models import File
from datetime import datetime

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile):
    try:
        # временно сохраняем файл
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # путь в S3
        s3_key = f"{datetime.now().strftime('%Y/%m/%d')}/{file.filename}"

        # заливаем в MinIO
        minio_client.fput_object(
            settings.MINIO_BUCKET,
            s3_key,
            temp_path
        )

        # добавляем запись в базу
        db = SessionLocal()
        db_file = File(
            filename=file.filename,
            file_type=file.content_type,
            s3_path=f"s3://{settings.MINIO_BUCKET}/{s3_key}",
            status="uploaded"
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        db.close()

        # чистим временный файл
        os.remove(temp_path)

        return {"status": "ok", "file_id": db_file.id, "path": db_file.s3_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
