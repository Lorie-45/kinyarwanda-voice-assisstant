from TTS.api import TTS
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
tts = None

@app.on_event("startup")
async def load_model():
    global tts
    try:
        logger.info("Loading TTS model...")
        # Using a multilingual model that should handle Kinyarwanda
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/your_tts",
            progress_bar=False,
            gpu=False
        )
        logger.info("TTS model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load TTS model: {str(e)}")
        raise

@app.get("/synthesize")
async def synthesize(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        output_file = "/tmp/output.wav"
        logger.info(f"Synthesizing: {text}")
        
        tts.tts_to_file(
            text=text,
            file_path=output_file,
            language="rw"  # Try 'rw' for Kinyarwanda or use a multilingual model
        )
        
        return FileResponse(
            output_file,
            media_type="audio/wav",
            filename="response.wav"
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)