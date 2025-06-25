"""
File Handler Utility for Financial Data Analysis AI Agent
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException

from app.config import settings
from .file_errors import FileError, FileErrorType, detect_file_error_type, create_user_friendly_error

logger = logging.getLogger(__name__)


class FileHandler:
    """
    File handling utility that manages uploads, storage, and cleanup
    Secure file management with session isolation
    """

    def __init__(self, upload_dir: Path = None):
        self.upload_dir = upload_dir or settings.UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile, session_id: str) -> Path:
        """
        Save uploaded file to session directory with detailed error handling
        """
        file_name = file.filename or "unknown_file"

        try:
            # Validate file with detailed error checking
            self._validate_file_detailed(file)

            # Create session directory
            session_dir = self.upload_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            # Generate safe filename
            safe_filename = self._generate_safe_filename(file.filename)
            file_path = session_dir / safe_filename

            # Check available disk space
            self._check_disk_space(file_path.parent)

            # Save file with error handling
            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()

                    # Check if file is empty
                    if len(content) == 0:
                        FileError.raise_http_exception(
                            FileErrorType.EMPTY_FILE,
                            file_name=file_name
                        )

                    buffer.write(content)
            except PermissionError as e:
                FileError.raise_http_exception(
                    FileErrorType.PERMISSION_ERROR,
                    additional_info=str(e),
                    file_name=file_name
                )
            except OSError as e:
                if "No space left" in str(e) or "disk full" in str(e).lower():
                    FileError.raise_http_exception(
                        FileErrorType.DISK_SPACE_ERROR,
                        additional_info=str(e),
                        file_name=file_name
                    )
                else:
                    FileError.raise_http_exception(
                        FileErrorType.UNKNOWN_ERROR,
                        additional_info=str(e),
                        file_name=file_name
                    )

            logger.info(f"File saved successfully: {file_path}")
            return file_path

        except HTTPException:
            # Re-raise HTTPExceptions (our detailed errors)
            raise
        except Exception as e:
            # Handle any unexpected errors
            logger.error(f"Unexpected file save error: {str(e)}")
            error_response = create_user_friendly_error(e, file_name)
            raise HTTPException(status_code=500, detail=error_response)

    def _validate_file_detailed(self, file: UploadFile):
        """Validate uploaded file with detailed error reporting"""
        file_name = file.filename or "unknown_file"

        # Check if filename exists
        if not file.filename:
            FileError.raise_http_exception(
                FileErrorType.INVALID_STRUCTURE,
                additional_info="No filename provided",
                file_name=file_name
            )

        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        supported_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet']

        if file_extension not in supported_extensions:
            FileError.raise_http_exception(
                FileErrorType.UNSUPPORTED_FORMAT,
                additional_info=f"File extension '{file_extension}' not supported. Supported: {', '.join(supported_extensions)}",
                file_name=file_name
            )

        # Check for macro-enabled Excel files
        if file_extension == '.xlsm':
            FileError.raise_http_exception(
                FileErrorType.UNSUPPORTED_FORMAT,
                additional_info="Macro-enabled Excel files (.xlsm) are not supported for security reasons",
                file_name=file_name
            )

        # Check content type if available
        if file.content_type:
            valid_content_types = [
                'text/csv',
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/json',
                'application/octet-stream'  # Generic binary
            ]

            if file.content_type not in valid_content_types:
                logger.warning(f"Unexpected content type: {file.content_type} for file: {file_name}")

    def _check_disk_space(self, directory: Path):
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(directory)

            # Require at least 100MB free space
            min_free_space = 100 * 1024 * 1024  # 100MB in bytes

            if free < min_free_space:
                FileError.raise_http_exception(
                    FileErrorType.DISK_SPACE_ERROR,
                    additional_info=f"Only {free // (1024*1024)}MB free space available, need at least 100MB"
                )
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

    def _validate_file(self, file: UploadFile):
        """Legacy validate method - kept for compatibility"""

        # Check file size
        if hasattr(file, 'size') and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )

        # Check file extension
        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in settings.ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
                )
        else:
            raise HTTPException(status_code=400, detail="No filename provided")

    def _generate_safe_filename(self, filename: str) -> str:
        """Generate safe filename"""
        if not filename:
            return f"upload_{uuid.uuid4().hex[:8]}.csv"

        # Keep original extension
        path = Path(filename)
        safe_name = "".join(c for c in path.stem if c.isalnum() or c in ('-', '_'))
        if not safe_name:
            safe_name = f"upload_{uuid.uuid4().hex[:8]}"

        return f"{safe_name}{path.suffix}"

    async def cleanup_session(self, session_id: str):
        """Cleanup session files"""
        try:
            session_dir = self.upload_dir / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)
                logger.info(f"Session {session_id} files cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed for session {session_id}: {str(e)}")

    def get_session_files(self, session_id: str) -> list:
        """Get list of files in session"""
        session_dir = self.upload_dir / session_id
        if session_dir.exists():
            return [f.name for f in session_dir.iterdir() if f.is_file()]
        return []

    def get_file_path(self, session_id: str, filename: str) -> Optional[Path]:
        """Get full path to file in session"""
        file_path = self.upload_dir / session_id / filename
        if file_path.exists():
            return file_path
        return None
