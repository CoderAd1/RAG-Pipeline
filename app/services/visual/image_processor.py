"""Image processing and description generation."""

from typing import Dict, Any, Optional
from loguru import logger
from app.utils.file_storage import storage


class ImageProcessor:
    """Process images and generate descriptions."""
    
    def process_image(
        self,
        image_data: Dict[str, Any],
        document_id: str,
        element_id: str,
        page_number: int
    ) -> Dict[str, Any]:
        """
        Process image and save to storage.
        
        Args:
            image_data: Raw image data from Docling
            document_id: Document ID
            element_id: Visual element ID
            page_number: Page number
            
        Returns:
            Processed image metadata
        """
        try:
            # Save image if PIL Image is available
            file_path = None
            if image_data.get("image_pil"):
                try:
                    # Convert PIL Image to bytes
                    from io import BytesIO
                    pil_image = image_data["image_pil"]
                    
                    logger.debug(f"Processing PIL image: {type(pil_image)}, size: {pil_image.size if pil_image else 'None'}")
                    
                    # Save as PNG
                    img_bytes = BytesIO()
                    pil_image.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    file_path = storage.save_image(
                        document_id=document_id,
                        element_id=element_id,
                        image_data=img_bytes.read(),
                        extension="png"
                    )
                    logger.info(f"Saved image to {file_path}")
                except Exception as img_error:
                    logger.error(f"Failed to save image: {img_error}")
            else:
                logger.warning(f"No PIL image data available for element {element_id}")
            
            # Get caption or generate description
            caption = image_data.get("caption", "")
            description = self._generate_description(caption, image_data)
            
            result = {
                "page_number": page_number,
                "element_type": self._classify_image_type(image_data),
                "file_path": file_path,
                "text_annotation": description,
                "bounding_box": image_data.get("bbox"),
                "metadata": {
                    "has_caption": bool(caption),
                    "original_caption": caption
                }
            }
            
            logger.debug(f"Processed image on page {page_number}, file_path: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return {
                "page_number": page_number,
                "element_type": "image",
                "file_path": None,
                "text_annotation": "Image (processing failed)",
                "metadata": {}
            }
    
    def _classify_image_type(self, image_data: Dict[str, Any]) -> str:
        """Classify image type based on available data."""
        caption = image_data.get("caption", "").lower()
        
        if "chart" in caption or "graph" in caption:
            return "chart"
        elif "figure" in caption or "fig" in caption:
            return "figure"
        else:
            return "image"
    
    def _generate_description(
        self,
        caption: str,
        image_data: Dict[str, Any]
    ) -> str:
        """
        Generate text description of image.
        
        In production, this would use a vision LLM. For now, use caption
        or create a basic description.
        """
        if caption:
            return f"Image: {caption}"
        
        # Basic description
        element_type = self._classify_image_type(image_data)
        bbox = image_data.get("bbox")
        
        description = f"{element_type.capitalize()}"
        if bbox:
            description += " (visual element)"
        
        return description
    
    def generate_vision_description(
        self,
        image_path: str,
        llm_service: Any
    ) -> str:
        """
        Generate vision LLM-based description of image.
        
        Args:
            image_path: Path to image file
            llm_service: LLM service with vision capabilities
            
        Returns:
            Generated description
        """
        prompt = "Describe this image in 1-2 sentences. Focus on the main content and any text visible."
        
        try:
            # This would call a vision model like GPT-4V or Claude with vision
            description = llm_service.generate_with_image(
                prompt=prompt,
                image_path=image_path,
                max_tokens=150
            )
            return description.strip()
        except Exception as e:
            logger.warning(f"Failed to generate vision description: {e}")
            return "Image content"
    
    def extract_image_text_ocr(self, image_data: bytes) -> str:
        """
        Extract text from image using OCR.
        
        This is a placeholder for OCR functionality.
        In production, integrate with Tesseract or cloud OCR services.
        """
        # Placeholder - would use pytesseract or similar
        logger.debug("OCR extraction not implemented")
        return ""
