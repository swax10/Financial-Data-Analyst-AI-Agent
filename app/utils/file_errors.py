"""
Detailed File Error Handling System
Provides specific error messages for different file upload issues
"""

from enum import Enum
from typing import Dict, Any, Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class FileErrorType(Enum):
    """Specific file error types"""
    ENCODING_ERROR = "encoding_error"
    CORRUPTED_FILE = "corrupted_file"
    PASSWORD_PROTECTED = "password_protected"
    FILE_TOO_LARGE = "file_too_large"
    UNSUPPORTED_FORMAT = "unsupported_format"
    EMPTY_FILE = "empty_file"
    INVALID_STRUCTURE = "invalid_structure"
    PERMISSION_ERROR = "permission_error"
    DISK_SPACE_ERROR = "disk_space_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


class FileError:
    """Detailed file error with user-friendly messages"""
    
    ERROR_MESSAGES = {
        FileErrorType.ENCODING_ERROR: {
            "title": "File Encoding Issue",
            "message": "Your file has encoding problems. This usually happens with non-UTF-8 files.",
            "solutions": [
                "Try saving your file as UTF-8 encoding",
                "Open in Excel and save as CSV (UTF-8)",
                "Use a text editor to convert encoding"
            ],
            "technical": "Character encoding detection failed or unsupported encoding"
        },
        FileErrorType.CORRUPTED_FILE: {
            "title": "Corrupted File",
            "message": "Your file appears to be corrupted or damaged.",
            "solutions": [
                "Try re-downloading the original file",
                "Open and re-save the file in Excel/LibreOffice",
                "Check if the file opens correctly on your computer"
            ],
            "technical": "File structure is invalid or contains corrupted data"
        },
        FileErrorType.PASSWORD_PROTECTED: {
            "title": "Password Protected File",
            "message": "Your Excel file is password protected and cannot be processed.",
            "solutions": [
                "Remove password protection from the Excel file",
                "Save as a new unprotected Excel file",
                "Export to CSV format instead"
            ],
            "technical": "Excel file requires password for access"
        },
        FileErrorType.FILE_TOO_LARGE: {
            "title": "File Too Large",
            "message": "Your file exceeds the maximum size limit.",
            "solutions": [
                "Split your data into smaller files",
                "Remove unnecessary columns or rows",
                "Compress the file or use a more efficient format"
            ],
            "technical": "File size exceeds configured maximum limit"
        },
        FileErrorType.UNSUPPORTED_FORMAT: {
            "title": "Unsupported File Format",
            "message": "This file format is not supported for analysis.",
            "solutions": [
                "Convert to CSV, Excel (.xlsx/.xls), or JSON format",
                "Supported formats: .csv, .xlsx, .xls, .json, .parquet",
                "Avoid macro-enabled files (.xlsm)"
            ],
            "technical": "File extension not in supported formats list"
        },
        FileErrorType.EMPTY_FILE: {
            "title": "Empty File",
            "message": "Your file appears to be empty or contains no data.",
            "solutions": [
                "Check that the file contains data",
                "Ensure the file isn't just headers without data rows",
                "Try uploading a different file"
            ],
            "technical": "File contains no readable data or only empty rows"
        },
        FileErrorType.INVALID_STRUCTURE: {
            "title": "Invalid File Structure",
            "message": "Your file structure doesn't match expected financial data format.",
            "solutions": [
                "Ensure your file has column headers",
                "Include at least one numeric column for amounts",
                "Check that data is properly formatted in rows and columns"
            ],
            "technical": "File structure validation failed - missing required elements"
        },
        FileErrorType.PERMISSION_ERROR: {
            "title": "File Permission Error",
            "message": "Cannot access the file due to permission restrictions.",
            "solutions": [
                "Close the file if it's open in another program",
                "Check file permissions on your system",
                "Try uploading from a different location"
            ],
            "technical": "Insufficient permissions to read or process the file"
        },
        FileErrorType.DISK_SPACE_ERROR: {
            "title": "Insufficient Storage Space",
            "message": "Not enough disk space to process your file.",
            "solutions": [
                "Try uploading a smaller file",
                "Contact support if this persists",
                "Compress your data before uploading"
            ],
            "technical": "Insufficient disk space for file processing"
        },
        FileErrorType.NETWORK_ERROR: {
            "title": "Network Upload Error",
            "message": "File upload was interrupted due to network issues.",
            "solutions": [
                "Check your internet connection",
                "Try uploading again",
                "Use a more stable network connection"
            ],
            "technical": "Network interruption during file upload"
        },
        FileErrorType.UNKNOWN_ERROR: {
            "title": "Unexpected Error",
            "message": "An unexpected error occurred while processing your file.",
            "solutions": [
                "Try uploading the file again",
                "Check that the file isn't corrupted",
                "Contact support if the problem persists"
            ],
            "technical": "Unhandled exception during file processing"
        }
    }
    
    @classmethod
    def create_error_response(cls, error_type: FileErrorType, 
                            additional_info: Optional[str] = None,
                            file_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a detailed error response"""
        
        error_info = cls.ERROR_MESSAGES.get(error_type, cls.ERROR_MESSAGES[FileErrorType.UNKNOWN_ERROR])
        
        response = {
            "success": False,
            "error_type": error_type.value,
            "error_title": error_info["title"],
            "error_message": error_info["message"],
            "solutions": error_info["solutions"],
            "technical_details": error_info["technical"]
        }
        
        if file_name:
            response["file_name"] = file_name
            
        if additional_info:
            response["additional_info"] = additional_info
            
        return response
    
    @classmethod
    def raise_http_exception(cls, error_type: FileErrorType, 
                           status_code: int = 400,
                           additional_info: Optional[str] = None,
                           file_name: Optional[str] = None):
        """Raise HTTPException with detailed error info"""
        
        error_response = cls.create_error_response(error_type, additional_info, file_name)
        
        # Log the technical details
        logger.error(f"File error ({error_type.value}): {error_response['technical_details']}")
        if additional_info:
            logger.error(f"Additional info: {additional_info}")
            
        raise HTTPException(
            status_code=status_code,
            detail=error_response
        )


def detect_file_error_type(exception: Exception, file_name: str = None) -> FileErrorType:
    """
    Detect the specific error type based on the exception
    """
    
    error_str = str(exception).lower()
    exception_type = type(exception).__name__.lower()
    
    # Encoding errors
    if any(keyword in error_str for keyword in ['encoding', 'codec', 'utf-8', 'decode', 'charmap']):
        return FileErrorType.ENCODING_ERROR
    
    # Password protection
    if any(keyword in error_str for keyword in ['password', 'encrypted', 'protected']):
        return FileErrorType.PASSWORD_PROTECTED
    
    # File size errors
    if any(keyword in error_str for keyword in ['too large', 'size limit', 'memory', 'out of memory']):
        return FileErrorType.FILE_TOO_LARGE
    
    # Corrupted file errors
    if any(keyword in error_str for keyword in ['corrupted', 'invalid', 'damaged', 'bad file', 'not a valid']):
        return FileErrorType.CORRUPTED_FILE
    
    # Empty file errors
    if any(keyword in error_str for keyword in ['empty', 'no data', 'no columns']):
        return FileErrorType.EMPTY_FILE
    
    # Permission errors
    if any(keyword in error_str for keyword in ['permission', 'access denied', 'forbidden']):
        return FileErrorType.PERMISSION_ERROR
    
    # Disk space errors
    if any(keyword in error_str for keyword in ['disk space', 'no space', 'storage']):
        return FileErrorType.DISK_SPACE_ERROR
    
    # Network errors
    if any(keyword in error_str for keyword in ['network', 'connection', 'timeout']):
        return FileErrorType.NETWORK_ERROR
    
    # Unsupported format (check file extension)
    if file_name:
        unsupported_extensions = ['.xlsm', '.doc', '.docx', '.pdf', '.txt']
        if any(file_name.lower().endswith(ext) for ext in unsupported_extensions):
            return FileErrorType.UNSUPPORTED_FORMAT
    
    # Exception type based detection
    if 'unicodeerror' in exception_type or 'decodeerror' in exception_type:
        return FileErrorType.ENCODING_ERROR
    elif 'permissionerror' in exception_type:
        return FileErrorType.PERMISSION_ERROR
    elif 'filenotfounderror' in exception_type:
        return FileErrorType.CORRUPTED_FILE
    elif 'memoryerror' in exception_type:
        return FileErrorType.FILE_TOO_LARGE
    
    return FileErrorType.UNKNOWN_ERROR


def create_user_friendly_error(exception: Exception, file_name: str = None) -> Dict[str, Any]:
    """
    Create a user-friendly error response from any exception
    """
    
    error_type = detect_file_error_type(exception, file_name)
    additional_info = f"Original error: {str(exception)}"
    
    return FileError.create_error_response(
        error_type=error_type,
        additional_info=additional_info,
        file_name=file_name
    )
