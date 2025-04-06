from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from models.chat import ChatRequest, ChatResponse
from services.llm_service import LLMService
from services.weather_service import WeatherService
from services.audio_service import AudioService
from services.chroma_service import ChromaService
from services.disease_service import DiseaseService
from config.settings import Settings
from utils.logging import setup_logging
import os

router = APIRouter()
settings = Settings()
logger = setup_logging()

chroma_service = ChromaService(settings)
llm_service = LLMService(settings.llm_model, chroma_service)
weather_service = WeatherService(settings)
audio_service = AudioService()
disease_service = DiseaseService(settings)

os.environ["PATH"] += os.pathsep + settings.ffmpeg_path

@router.post("/chat", response_model=ChatResponse)
async def chat_text(
    question: str = Form(...),
    conversation_history: Optional[List[str]] = Form(None),
    detected_disease: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        # Process image if provided, overriding detected_disease from form-data
        if image:
            image_content = await image.read()
            logger.info(f"Received image: {image.filename}, size: {len(image_content)} bytes")
            try:
                detected_disease = disease_service.process_uploaded_image(image_content, image.filename)
                logger.info(f"Disease detection result: {detected_disease}")
            except Exception as e:
                logger.error(f"Error in disease detection: {str(e)}")
                detected_disease = None  # Proceed without disease if detection fails

        # Create request object
        request = ChatRequest(
            question=question,
            conversation_history=conversation_history or [],
            detected_disease=detected_disease,
            country=country
        )
        
        # Log the request for debugging
        logger.info(f"Processing request with detected_disease: {request.detected_disease}")
        
        return await llm_service.process_query(request, weather_service)
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/audio")
async def chat_audio(
    audio_file: UploadFile = File(...),
    detected_disease: Optional[str] = Form(None),
    conversation_history: Optional[List[str]] = Form(None),
    country: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    logger.info("Received audio request")
    tmp_audio_path = None
    output_path = "response.mp3"
    
    try:
        tmp_audio_path = audio_service.validate_audio(await audio_file.read(), audio_file.content_type)
        question = audio_service.speech_to_text(tmp_audio_path)

        if image:
            image_content = await image.read()
            detected_disease = disease_service.process_uploaded_image(image_content, image.filename)

        chat_request = ChatRequest(
            question=question,
            conversation_history=conversation_history or [],
            detected_disease=detected_disease,
            country=country
        )

        llm_response = await llm_service.process_query(chat_request, weather_service)
        audio_base64 = audio_service.text_to_speech(llm_response.response, output_path)

        return {
            "question": question,
            "text_response": llm_response.response,
            "audio_base64": audio_base64
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.unlink(tmp_audio_path)
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")