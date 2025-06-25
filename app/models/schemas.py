"""
Pydantic models for Financial Data Analysis AI Agent
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.now)


class FileUploadResponse(BaseModel):
    """Response for file upload"""
    session_id: str
    filename: str
    file_size: int
    file_type: str
    columns: List[str]
    row_count: int
    data_summary: Dict[str, Any]
    success: bool
    message: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Request for data analysis"""
    session_id: str
    query: str
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


class ExecutionResult(BaseModel):
    """Result of code execution"""
    success: bool
    results: List[Any] = Field(default_factory=list)  # Allow Result objects or dicts
    logs: Dict[str, List[str]] = Field(default_factory=lambda: {"stdout": [], "stderr": []})
    error: Optional[Dict[str, str]] = None
    execution_count: Optional[int] = None
    execution_time: Optional[float] = None


class AnalysisResponse(BaseModel):
    """Response for data analysis"""
    session_id: str
    query: str
    generated_code: str
    execution_result: ExecutionResult
    explanation: str
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class OutputMessage(BaseModel):
    """Output message from code execution"""
    line: str
    timestamp: int
    error: bool = False


class Result(BaseModel):
    """Execution result data"""
    text: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    svg: Optional[str] = None
    png: Optional[str] = None
    jpeg: Optional[str] = None
    pdf: Optional[str] = None
    latex: Optional[str] = None
    json: Optional[Dict] = None
    javascript: Optional[str] = None
    data: Optional[Dict] = None
    is_main_result: bool = False
    extra: Optional[Dict] = None


class ExecutionError(BaseModel):
    """Execution error"""
    name: str
    value: str
    traceback: str


class Context(BaseModel):
    """Execution context"""
    id: str
    language: str
    cwd: str


class DataSummary(BaseModel):
    """Summary of uploaded data"""
    file_type: str
    encoding: Optional[str] = None
    columns: List[str]
    column_types: Dict[str, str]
    row_count: int
    file_size: int
    has_header: bool
    date_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    missing_values: Dict[str, int] = Field(default_factory=dict)
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)


class FinancialPattern(BaseModel):
    """Detected financial patterns in data"""
    expense_columns: List[str] = Field(default_factory=list)
    income_columns: List[str] = Field(default_factory=list)
    date_columns: List[str] = Field(default_factory=list)
    category_columns: List[str] = Field(default_factory=list)
    amount_columns: List[str] = Field(default_factory=list)
    balance_columns: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0


class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    filename: str
    file_path: str
    created_at: datetime
    last_activity: datetime
    query_count: int = 0
    data_summary: Optional[DataSummary] = None
    financial_patterns: Optional[FinancialPattern] = None


class StreamingChunk(BaseModel):
    """Streaming response chunk"""
    type: str  # "code", "execution", "result", "error", "complete"
    content: Union[str, Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)


class CodeGenerationRequest(BaseModel):
    """Request for code generation"""
    query: str
    data_context: Dict[str, Any]
    file_path: str
    language: str = "python"
    include_visualization: bool = True


class CodeGenerationResponse(BaseModel):
    """Response for code generation"""
    code: str
    explanation: str
    imports: List[str]
    functions_used: List[str]
    visualization_type: Optional[str] = None
    confidence_score: float = 0.0
