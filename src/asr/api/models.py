"""
Pydantic models cho API request/response
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class TranscriptionRequest(BaseModel):
    """
    Request model cho transcription
    """
    # URL của audio file hoặc base64 encoded audio
    audio_url: Optional[str] = Field(None, description="URL đến audio file")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio data")
    
    # Parameters
    language: str = Field("vi", description="Ngôn ngữ (vi: tiếng Việt)")
    task: str = Field("transcribe", description="Task: transcribe hoặc translate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_url": "https://example.com/audio.wav",
                "language": "vi",
                "task": "transcribe"
            }
        }


class TranscriptionResponse(BaseModel):
    """
    Response model cho transcription
    """
    success: bool = Field(True, description="Trạng thái thành công")
    text: Optional[str] = Field(None, description="Văn bản được transcribe")
    confidence: Optional[float] = Field(None, description="Độ tin cậy")
    processing_time: Optional[float] = Field(None, description="Thời gian xử lý (giây)")
    error: Optional[str] = Field(None, description="Thông tin lỗi nếu có")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata khác")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "text": "Xin chào Việt Nam",
                "confidence": 0.95,
                "processing_time": 1.23
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str = "healthy"
    model_loaded: bool = True
    model_name: Optional[str] = None
    device: Optional[str] = None
