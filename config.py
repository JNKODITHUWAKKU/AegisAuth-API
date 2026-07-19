# config.py — Environment-based configuration management
# All settings are driven by environment variables with sensible defaults.
# Copy .env.example to .env and override as needed.

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables / .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Application ─────────────────────────────────────────────────────────
    APP_NAME: str = "AegisAuth API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Industry-grade Face Verification & Anti-Spoofing Microservice. "
        "Plug-and-play with any Lecture Attendance or Examination Portal."
    )
    DEBUG: bool = False

    # ── Model Paths ──────────────────────────────────────────────────────────
    YOLO_FACE_MODEL_PATH: str = Field(default="models/yolov8n-face.pt")
    LIVENESS_MODEL_PATH: str = Field(default="models/liveness_engine/weights/best.pt")

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_DIR: str = Field(default="./database/identities")

    # ── Thresholds ───────────────────────────────────────────────────────────
    LIVENESS_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)
    MATCH_TOLERANCE: float = Field(default=0.85, ge=0.0, le=2.0)

    # ── API Security ─────────────────────────────────────────────────────────
    # Set this in your .env. External systems must send this key in
    # the "X-API-Key" header for every request.
    API_KEY: str = Field(default="change-me-in-production")

    # ── Image Constraints ────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024   # 10 MB hard limit
    MIN_FACE_SIZE_PX: int = 40                       # Reject tiny faces

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE: str = Field(default="cpu")               # "cpu" | "cuda"


# Singleton instance — import this everywhere
settings = Settings()
