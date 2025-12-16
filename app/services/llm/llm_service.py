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


class LLMService:
    """Service for LLM-based answer generation using Groq."""

    def __init__(self, model: Optional[str] = None):
        """
        Initialize LLM service with Groq.

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
                        "content": (
                            "You are an expert data analyst and research assistant. "
                            "Answer questions by PRIORITIZING the provided document context (text, tables, figures), "
                            "but you may also use your general knowledge when:\n"
                            "  - The context is incomplete or doesn't fully answer the question\n"
                            "  - The question asks for well-known facts that aren't in the context\n"
                            "  - Additional context from your knowledge enhances understanding\n\n"
                            "When using context: Extract specific values accurately, cite sources with [Source N].\n"
                            "When using general knowledge: Clearly state 'Based on general knowledge' or similar.\n"
                            "Combine both intelligently for comprehensive answers."
                        )
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
        """Create enhanced prompt for answer generation with multimodal awareness."""
        prompt = f"""Document Context (including tables, images, charts, and figures):
{context}

Question: {question}

ANSWER INSTRUCTIONS:

1. VISUAL ELEMENT DETECTION:
   - Look for [Source N] entries that mention "table", "chart", "graph", "figure", "image"
   - These represent visual elements from the document

2. VISUAL CONTENT INTEGRATION:
   - When answering, reference visual elements by their descriptions
   - Explain what charts/graphs show and how they relate to the question
   - If a table is referenced, extract specific data points mentioned
   - Describe trends, patterns, or insights shown in visual elements

3. TABLE DATA EXTRACTION (CRITICAL):
   - TABLES ARE IN MARKDOWN FORMAT (with | symbols):
     | Header1    | Header2   | Header3  |
     |------------|-----------|----------|
     | Row1Value1 | Row1Value2| Row1Value3|

   - HOW TO EXTRACT: Find correct ROW by first column, then COLUMN by headers
   - Extract EXACT values from row × column intersections
   - Match names case-insensitively, handle variations

4. CHART/GRAPH ANALYSIS:
   - Describe what the visualization shows
   - Explain trends, comparisons, relationships depicted
   - Connect visual insights to text content

5. CITATION REQUIREMENTS:
   - Cite ALL sources: [Source N, Page X]
   - Specify if information comes from text, table, or visual element
   - Example: "The bar chart on page 5 shows..." or "According to the table on page 3..."

6. ANSWER STRUCTURE FOR VISUAL QUERIES:
   - If question asks "show me" or "what does X look like":
     → Describe the relevant visual element in detail
     → Explain what it shows and key insights
   - If question is about data/trends:
     → Reference both tabular data and visual representations
     → Explain how visuals illustrate the data

7. ACCURACY FIRST:
   - Extract EXACT values from tables/visuals
   - Don't approximate unless explicitly stated
   - If visual shows a clear trend/pattern, describe it accurately

8. HANDLING INCOMPLETE CONTEXT:
   - If the context doesn't fully answer the question, use your general knowledge
   - Clearly distinguish between context-based and knowledge-based information
   - Provide comprehensive answers by intelligently combining both sources

Answer comprehensively using the context AND your knowledge as needed:"""

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
