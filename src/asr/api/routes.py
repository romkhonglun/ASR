"""
FastAPI routes
"""
import base64
import tempfile
import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from .models import TranscriptionRequest, TranscriptionResponse, HealthResponse
from .inference import ASRInference

router = APIRouter()

# Global inference engine
inference_engine: Optional[ASRInference] = None


def get_inference_engine(model_path: str = None) -> ASRInference:
    """Get hoặc tạo inference engine"""
    global inference_engine
    if inference_engine is None:
        inference_engine = ASRInference(model_path=model_path)
    return inference_engine


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcribe audio từ URL hoặc base64
    """
    engine = get_inference_engine()
    
    # Xử lý audio input
    audio_input = None
    
    if request.audio_url:
        audio_input = request.audio_url
    elif request.audio_base64:
        # Decode base64 và tạo temp file
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_input = tmp.name
        except Exception as e:
            return TranscriptionResponse(
                success=False,
                error=f"Error decoding base64: {str(e)}"
            )
    else:
        return TranscriptionResponse(
            success=False,
            error="Either audio_url or audio_base64 must be provided"
        )
    
    # Transcribe
    result = engine.transcribe(
        audio_input=audio_input,
        language=request.language,
        task=request.task
    )
    
    # Cleanup temp file nếu có
    if request.audio_base64 and os.path.exists(audio_input):
        os.unlink(audio_input)
    
    return TranscriptionResponse(
        success=result.get("success", False),
        text=result.get("text"),
        confidence=result.get("confidence"),
        processing_time=result.get("processing_time"),
        error=result.get("error"),
        metadata=result.get("metadata")
    )


@router.post("/transcribe-file", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    task: str = Form("transcribe")
):
    """
    Transcribe audio file được upload lên
    """
    # Kiểm tra file type
    allowed_content_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/flac", "audio/x-wav"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type không được hỗ trợ. Chỉ chấp nhận: {', '.join(allowed_content_types)}"
        )
    
    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe
        engine = get_inference_engine()
        result = engine.transcribe(
            audio_input=tmp_path,
            language=language,
            task=task
        )
        
        return TranscriptionResponse(
            success=result.get("success", False),
            text=result.get("text"),
            confidence=result.get("confidence"),
            processing_time=result.get("processing_time"),
            error=result.get("error"),
            metadata=result.get("metadata")
        )
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    engine = get_inference_engine()
    model_info = engine.get_model_info()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_info.get("model_loaded", False),
        model_name=model_info.get("model_path"),
        device=model_info.get("device")
    )


@router.get("/model-info")
async def model_info():
    """
    Lấy thông tin chi tiết về model
    """
    engine = get_inference_engine()
    return engine.get_model_info()
