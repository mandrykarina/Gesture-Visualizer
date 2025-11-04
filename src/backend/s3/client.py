from minio import Minio
from config import settings

minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE
)

# создаём бакет, если его нет
if not minio_client.bucket_exists(settings.MINIO_BUCKET):
    minio_client.make_bucket(settings.MINIO_BUCKET)
