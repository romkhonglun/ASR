"""
Script chạy API server
"""
import argparse
import uvicorn
import os


def main():
    parser = argparse.ArgumentParser(description="Start ASR API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=None, help="Path to fine-tuned model")
    
    args = parser.parse_args()
    
    # Set environment variable cho model path
    if args.model_path:
        os.environ["ASR_MODEL_PATH"] = args.model_path
    
    # Set ASR module path
    app_path = "asr.api.app:create_app"
    
    print(f"Starting ASR API server on {args.host}:{args.port}")
    print(f"Model path: {args.model_path or 'vinai/PhoWhisper-small'}")
    print(f"Workers: {args.workers}")
    print(f"Auto-reload: {args.reload}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app_path,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
