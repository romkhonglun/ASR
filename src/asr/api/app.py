"""
FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


def create_app() -> FastAPI:
    """
    Tạo FastAPI application
    """
    app = FastAPI(
        title="ASR Service - Vietnamese Speech Recognition",
        description="""
        REST API cho nhận dạng giọng nói tiếng Việt (ASR).
        
        ## Tính năng
        - **Transcription**: Chuyển đổi audio sang văn bản
        - **File Upload**: Upload audio file để transcribe
        - **Health Check**: Kiểm tra tình trạng service
        
        ## Supported Formats
        - WAV
        - MP3
        - FLAC
        
        ## Model
        Fine-tuned trên dataset VLSP 2020
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1", tags=["ASR"])
    
    @app.get("/")
    async def root():
        return {
            "message": "ASR Service - Vietnamese Speech Recognition API",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    return app
