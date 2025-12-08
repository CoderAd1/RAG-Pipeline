"""LLM service for answer generation using Groq and Gemini."""

from typing import List, Dict, Any, Optional
from loguru import logger
from app.core.config import settings

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available")


class LLMService:
    """Service for LLM-based answer generation using Groq (text) and Gemini (vision)."""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize LLM service with Groq for text and Gemini for vision.
        
        Args:
            model: Model name (defaults to settings.default_llm_model)
        """
        # Initialize Groq for text generation
        if not GROQ_AVAILABLE:
            raise RuntimeError(
                "Groq not installed. Install with: pip install groq"
            )
        
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.model_name = model or settings.default_llm_model
        
        # Initialize Gemini for vision tasks
        self.gemini_available = GEMINI_AVAILABLE
        if GEMINI_AVAILABLE and settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            logger.info("Gemini configured for vision tasks")
        
        logger.info(f"LLM service initialized: Groq - {self.model_name}")
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        visual_elements: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate answer based on question and retrieved context.
        
        Args:
            question: User question
            context_chunks: Retrieved text chunks
            visual_elements: Retrieved visual elements (tables/images)
            
        Returns:
            Generated answer
        """
        # Build context
        context = self._build_context(context_chunks, visual_elements)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        # Generate answer using Groq
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant answering questions based on provided document context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            
            answer = response.choices[0].message.content
            logger.debug(f"Generated answer using Groq ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
    
    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
        visual_elements: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build context string from chunks and visual elements."""
        context_parts = []
        
        # Add text chunks
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            page = chunk.get("page", "unknown")
            doc_name = chunk.get("document_name", "document")
            
            context_parts.append(
                f"[Source {i}] (from {doc_name}, page {page}):\n{text}"
            )
        
        # Add visual elements if present
        if visual_elements:
            for i, element in enumerate(visual_elements, len(chunks) + 1):
                element_type = element.get("element_type", "visual")
                description = element.get("description", "")
                page = element.get("page_number", "unknown")
                
                # Add table markdown if available
                if element_type == "table" and element.get("table_markdown"):
                    context_parts.append(
                        f"[Source {i}] ({element_type} on page {page}):\n"
                        f"{element.get('table_markdown')}\n"
                        f"Description: {description}"
                    )
                else:
                    context_parts.append(
                        f"[Source {i}] ({element_type} on page {page}):\n{description}"
                    )
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for answer generation."""
        prompt = f"""Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite sources using [Source N] notation
- Be concise but comprehensive
- If tables or visual elements are referenced, incorporate that information

Answer:"""
        
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt (generic method).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
    
    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text with image input (for vision tasks).
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            from PIL import Image
            
            # Load image
            image = Image.open(image_path)
            
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens or settings.llm_max_tokens,
            )
            
            # Use Gemini 2.0 Flash (supports vision)
            vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            response = vision_model.generate_content(
                [prompt, image],
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini vision generation failed: {e}")
            raise
