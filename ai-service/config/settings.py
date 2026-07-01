from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    postgres_user: str = "aiobs_user"
    postgres_password: str = "aiobs_pass"
    postgres_db: str = "aiobs_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    redis_host: str = "localhost"
    redis_port: int = 6379

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    models_dir: str = "./models"
    model_version: str = "v2.0.0"

    alert_webhook_url: Optional[str] = None
    alert_email_enabled: bool = False
    slack_webhook_url: Optional[str] = None

    drift_psi_threshold: float = 0.50
    drift_kl_threshold: float = 0.15
    drift_check_interval: int = 60

    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
