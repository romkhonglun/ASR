"""
ASR Inference module - xử lý transcription
"""
import time
import os
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class ASRInference:
    """
    Module inference cho ASR
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        torch_dtype=None
    ):
        """
        Khởi tạo ASR inference
        
        Args:
            model_path: Đường dẫn đến model đã fine-tune hoặc model name
            device: cuda/cpu
            torch_dtype: torch dtype cho model
        """
        self.model_path = model_path or os.getenv("ASR_MODEL_PATH", "vinai/PhoWhisper-small")
        
        # Tự động chọn device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        
        self.model = None
        self.processor = None
        self.pipe = None
        
    def load_model(self):
        """Load model và processor"""
        print(f"Loading model từ: {self.model_path}")
        print(f"Device: {self.device}")
        
        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        
        self.model.to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Tạo pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        print("Model loaded successfully!")
        return self.pipe
    
    def transcribe(
        self,
        audio_input,
        language: str = "vi",
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio
        
        Args:
            audio_input: Audio file path, URL, hoặc numpy array
            language: Ngôn ngữ (vi, en, ...)
            task: transcribe hoặc translate
            **kwargs: Additional arguments
            
        Returns:
            Dict chứa kết quả transcription
        """
        if self.pipe is None:
            self.load_model()
            
        start_time = time.time()
        
        try:
            # Chạy inference
            result = self.pipe(
                audio_input,
                generate_kwargs={
                    "language": language,
                    "task": task
                },
                return_timestamps=True,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "text": result.get("text", ""),
                "chunks": result.get("chunks", None),
                "processing_time": processing_time,
                "metadata": {
                    "language": language,
                    "task": task,
                    "device": self.device,
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "text": None,
                "error": str(e),
                "processing_time": processing_time,
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Lấy thông tin model"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "model_loaded": self.model is not None,
        }
