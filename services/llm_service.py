from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from services.chroma_service import ChromaService
from services.weather_service import WeatherService
from models.chat import ChatRequest, ChatResponse
from utils.logging import setup_logging

logger = setup_logging()

class LLMService:
    def __init__(self, model_name: str, chroma_service: ChromaService):
        self.model = OllamaLLM(model=model_name)
        self.chroma_service = chroma_service
        self.prompt_template = """
You are an expert in agriculture, specializing in detecting plant diseases, their causes, symptoms, treatments, and prevention strategies. 
You also advise on the best plants to grow based on weather conditions (temperature, humidity) and provide reasoning for your recommendations to maximize benefits like yield and resilience.

IMPORTANT: When a plant disease is detected in an uploaded image, format your response in Markdown with the following structure:
- Use `## Detected Disease: [disease_name]` as a heading.
- Use `### Symptoms`, `### Causes`, `### Treatment Options`, and `### Prevention Strategies` as subheadings.
- Use bold (`**text**`) for emphasis and bullet points (`- `) for lists.
- Provide comprehensive information under each section.

Detected Disease: {detected_disease}
If a disease has been detected in the uploaded image, provide comprehensive information about this specific disease following the Markdown structure above. Be very thorough in your explanation and recommendations.

Conversation History:
{conversation_history}

Context from Memory:
{context}

Weather Data (if applicable):
{weather_data}

User Question: {question}
Assistant:
"""
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

    async def process_query(self, request: ChatRequest, weather_service: WeatherService) -> ChatResponse:
        try:
            # Get disease-specific context if available
            context = ""
            if request.detected_disease:
                logger.info(f"Getting context for detected disease: {request.detected_disease}")
                disease_info = self.chroma_service.retrieve_context(request.detected_disease)
                context += f"Disease Information: {disease_info}\n\n"
            
            # Get context for the question
            question_context = self.chroma_service.retrieve_context(request.question)
            context += question_context
            
            # Get weather data if country is provided
            weather_data = await weather_service.fetch_weather_summary(request.country) if request.country else ""

            # Format conversation history
            conversation_history = "\n".join(request.conversation_history) if request.conversation_history else ""
            
            # Format prompt input
            formatted_input = self.prompt.format(
                question=request.question,
                conversation_history=conversation_history,
                context=context,
                weather_data=weather_data,
                detected_disease=request.detected_disease or "No disease detected in the image"
            )
            
            # Log for debugging
            logger.info(f"Sending prompt with detected_disease: {request.detected_disease}")
            
            # Get response from LLM
            response_text = self.model.invoke(formatted_input)
            
            # Store conversation
            self.chroma_service.store_conversation(request.question, response_text)
            
            return ChatResponse(response=response_text)
        except Exception as e:
            logger.error(f"LLM processing error: {str(e)}")
            raise