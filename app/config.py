"""
Configuration settings for Financial Data Analysis AI Agent
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Application settings
    APP_NAME: str = "Financial Data Analysis AI Agent"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # File handling limits
    UPLOAD_DIR: Path = Path("workspace/uploads")
    MAX_FILE_SIZE: int = 512 * 1024 * 1024  # 512MB (same as OpenAI)
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls", ".json", ".parquet"]

    # LLM settings
    LLM_MODEL_PATH: str = "C:\\models\\Llama-xLAM-2-8B-fc-r-Q8_0.gguf"
    LLM_CONTEXT_LENGTH: int = 4096  # Increased for multi-sheet Excel support
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    # Code execution settings
    EXECUTION_TIMEOUT: int = 300  # 5 minutes
    MAX_MEMORY_MB: int = 2048  # 2GB
    PYTHON_PATH: str = "python"
    JUPYTER_PORT: int = 8888

    # Security settings
    ENABLE_SANDBOX: bool = True
    ALLOWED_IMPORTS: List[str] = [
        "pandas", "numpy", "matplotlib", "matplotlib.pyplot", "seaborn", "plotly",
        "sklearn", "scipy", "openpyxl", "xlrd", "json", "csv",
        "datetime", "math", "statistics", "re", "os", "pathlib"
    ]

    # Session management
    SESSION_TIMEOUT: int = 3600  # 1 hour
    MAX_SESSIONS: int = 100

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Financial analysis specific
    FINANCIAL_KEYWORDS: List[str] = [
        "expense", "income", "revenue", "profit", "loss", "budget",
        "transaction", "payment", "balance", "amount", "cost",
        "price", "value", "total", "sum", "average", "trend",
        "category", "date", "month", "year", "quarterly", "annual"
    ]

    # Data processing
    CHUNK_SIZE: int = 10000  # For large file processing
    ENCODING_DETECTION: bool = True
    AUTO_TYPE_INFERENCE: bool = True

    # Execution environment
    DEFAULT_TEMPLATE: str = "financial-data-analyst"
    DEFAULT_TIMEOUT: int = 60  # seconds
    REQUEST_TIMEOUT: int = 30  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


# Create settings instance
settings = Settings()

# Ensure directories exist (workspace structure)
def ensure_directories():
    """Create necessary directories"""
    directories = [
        settings.UPLOAD_DIR,
        Path("workspace/sessions"),
        Path("workspace/outputs"),
        Path("workspace/temp"),
        Path("logs"),
        Path("models")
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
ensure_directories()
