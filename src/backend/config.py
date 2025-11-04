from pydantic import BaseSettings

class Settings(BaseSettings):
    POSTGRES_USER: str = "gesture_user"
    POSTGRES_PASSWORD: str = "gesture_pass"
    POSTGRES_DB: str = "gesture_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "gesture-data"
    MINIO_SECURE: bool = False  # http, не https

    @property
    def DATABASE_URL(self):
        # формируем строку подключения для SQLAlchemy
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
