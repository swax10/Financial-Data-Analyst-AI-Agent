"""
Financial Data Analysis AI Agent - Main FastAPI Application
"""

import logging
import sys
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    FileUploadResponse,
    HealthResponse
)
from app.services.data_processor import DataProcessor
from app.services.llm_service import LLMService
from app.services.code_executor import CodeExecutor
from app.services.response_formatter import ResponseFormatter
from app.utils.file_handler import FileHandler
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Global services
data_processor: Optional[DataProcessor] = None
llm_service: Optional[LLMService] = None
code_executor: Optional[CodeExecutor] = None
response_formatter: Optional[ResponseFormatter] = None
file_handler: Optional[FileHandler] = None

# Active sessions storage
active_sessions: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global data_processor, llm_service, code_executor, response_formatter, file_handler

    logger.info("Starting Financial Data Analysis AI Agent...")

    # Initialize services
    data_processor = DataProcessor()
    llm_service = LLMService(model_path=settings.LLM_MODEL_PATH)
    code_executor = CodeExecutor()
    file_handler = FileHandler(upload_dir=settings.UPLOAD_DIR)

    # Initialize LLM first, then pass it to response formatter for dual LLM approach
    logger.info("=== STARTUP: Initializing LLM service...")
    await llm_service.initialize()
    logger.info("=== STARTUP: LLM service initialized successfully!")

    # Initialize remaining services
    logger.info("=== STARTUP: Initializing code executor...")
    await code_executor.initialize()
    logger.info("=== STARTUP: Code executor initialized successfully!")

    # Set global variables (including response formatter with LLM service)
    logger.info("=== STARTUP: Creating response formatter with LLM service...")
    global response_formatter
    response_formatter = ResponseFormatter(llm_service=llm_service)
    logger.info(f"=== STARTUP: Response formatter created! Type: {type(response_formatter)}")
    logger.info(f"=== STARTUP: Response formatter has LLM service: {response_formatter.llm_service is not None}")
    logger.info(f"=== STARTUP: LLM service type: {type(response_formatter.llm_service)}")

    logger.info("All services initialized successfully")
    yield

    # Cleanup
    logger.info("Shutting down services...")
    if code_executor:
        await code_executor.cleanup()

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Financial Data Analysis AI Agent",
    description="AI-powered financial data analysis with flexible CSV/Excel processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        services={
            "data_processor": data_processor is not None,
            "llm_service": llm_service is not None,
            "code_executor": code_executor is not None,
            "file_handler": file_handler is not None
        }
    )


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Upload and process financial data file"""
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(f"Processing file upload: {file.filename} for session {session_id}")

        # Save uploaded file
        file_path = await file_handler.save_upload(file, session_id)

        # Process and analyze file
        analysis_result = await data_processor.analyze_file(file_path)

        # Store session data
        active_sessions[session_id] = {
            "file_path": str(file_path),
            "file_info": analysis_result,
            "created_at": asyncio.get_event_loop().time()
        }

        # Include financial patterns in the response for schema detection
        response_data = analysis_result.copy()
        response_data["financial_patterns"] = analysis_result.get("financial_patterns", {})

        return FileUploadResponse(
            session_id=session_id,
            filename=file.filename,
            file_size=analysis_result["file_size"],
            file_type=analysis_result["file_type"],
            columns=analysis_result["columns"],
            row_count=analysis_result["row_count"],
            data_summary=response_data,  # Include financial patterns
            success=True
        )

    except HTTPException:
        # Re-raise HTTP exceptions with their original status codes
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Analyze financial data with natural language query"""
    try:
        # Validate session
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = active_sessions[request.session_id]
        file_path = session_data["file_path"]
        file_info = session_data["file_info"]

        logger.info(f"Processing analysis request for session {request.session_id}")
        logger.info(f"Query: {request.query}")

        # Generate analysis code using LLM
        code_generation_result = await llm_service.generate_analysis_code(
            query=request.query,
            file_info=file_info,
            file_path=file_path
        )

        # Execute generated code
        execution_result = await code_executor.execute_code(
            code=code_generation_result["code"],
            session_id=request.session_id
        )

        # Generate human-friendly response using dual LLM approach
        logger.info("=== MAIN: Calling response formatter with dual LLM approach...")
        logger.info(f"=== Response formatter type: {type(response_formatter)}")
        logger.info(f"=== Response formatter has LLM service: {hasattr(response_formatter, 'llm_service') and response_formatter.llm_service is not None}")

        
        print("=== MAIN: About to call response formatter")
        
        print(f"📝 Query: {request.query}")
        print(f"✅ Execution success: {execution_result.success}")
        print(f"💻 Code length: {len(code_generation_result['code'])}")
        print(f"🎯 Response formatter available: {response_formatter is not None}")

        human_response = await response_formatter.format_analysis_response(
            request.query,
            execution_result.dict(),
            code_generation_result["code"]
        )

        
        print("=== MAIN: Response formatter completed")
        
        print(f"📝 Human response length: {len(human_response)}")
        print(f"📝 Human response preview: {human_response[:200]}...")
        logger.info("=== MAIN: Response formatting completed successfully")

        # Update explanation with human-friendly response
        enhanced_explanation = f"{human_response}\n\n---\n\n**Technical Details:**\n{code_generation_result['explanation']}"

        return AnalysisResponse(
            session_id=request.session_id,
            query=request.query,
            generated_code=code_generation_result["code"],
            execution_result=execution_result,
            explanation=enhanced_explanation,
            success=execution_result.success
        )

    except HTTPException:
        # Re-raise HTTP exceptions with their original status codes
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/query")
async def query_data(request: AnalysisRequest):
    """Query financial data with natural language (Assignment requirement: /query endpoint)"""
    # This is an alias for the analyze endpoint to meet assignment requirements
    return await analyze_data(request)


@app.get("/stream-analysis/{session_id}")
async def stream_analysis(session_id: str, query: str):
    """Stream real-time analysis results"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        async def generate_stream():
            session_data = active_sessions[session_id]
            file_path = session_data["file_path"]
            file_info = session_data["file_info"]

            # Stream code generation
            async for chunk in llm_service.stream_code_generation(
                query=query,
                file_info=file_info,
                file_path=file_path
            ):
                yield f"data: {chunk}\n\n"

            # Execute and stream results
            # Implementation continues...

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )

    except Exception as e:
        logger.error(f"Streaming analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "filename": Path(data["file_path"]).name,
                "created_at": data["created_at"]
            }
            for sid, data in active_sessions.items()
        ]
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup files"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = active_sessions[session_id]

        # Cleanup files
        await file_handler.cleanup_session(session_id)

        # Remove from active sessions
        del active_sessions[session_id]

        return {"message": f"Session {session_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Session deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
