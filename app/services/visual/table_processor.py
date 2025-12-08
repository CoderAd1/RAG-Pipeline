"""Table processing and description generation."""

from typing import Dict, Any, Optional
import json
from loguru import logger
from app.utils.file_storage import storage


class TableProcessor:
    """Process tables and generate descriptions."""
    
    def __init__(self):
        """Initialize table processor."""
        self.storage = storage
    
    def process_table(
        self,
        table_data: dict,
        document_id: str,
        element_id: str,
        page_number: int,
        pdf_path: Optional[str] = None
    ) -> dict:
        """
        Process table data and save to storage.
        
        Args:
            table_data: Dictionary with table information
            document_id: Document ID
            element_id: Visual element ID
            page_number: Page number
            pdf_path: Optional path to PDF for image extraction
            
        Returns:
            Processed table metadata
        """
        # Extract table image if PDF path and bbox are provided
        file_path = None
        
        # DEBUG: Write to file to verify execution
        with open("/tmp/table_debug.txt", "a") as f:
            f.write(f"process_table called: pdf_path={pdf_path is not None}, bbox={table_data.get('bbox') is not None}\n")
        
        logger.debug(f"Table processing - pdf_path: {pdf_path is not None}, bbox: {table_data.get('bbox') is not None}")
        if pdf_path and table_data.get('bbox'):
            try:
                logger.info(f"Attempting to extract table image with bbox: {table_data['bbox']}")
                file_path = self._extract_table_image(
                    pdf_path=pdf_path,
                    bbox=table_data['bbox'],
                    page_number=page_number,
                    document_id=document_id,
                    element_id=element_id
                )
                logger.info(f"Extracted table image: {file_path}")
            except Exception as e:
                import traceback
                logger.error(f"Failed to extract table image: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            if not pdf_path:
                logger.warning("No PDF path provided for table image extraction")
            if not table_data.get('bbox'):
                logger.warning(f"No bbox in table_data. Keys: {list(table_data.keys())}")
        
        # Get markdown representation
        markdown = table_data.get("markdown", table_data.get("text", ""))
        
        # Save table data as JSON
        table_json_path = self.storage.save_table_json(
            document_id=document_id,
            element_id=element_id,
            table_data={
                "markdown": markdown,
                "page": page_number,
                "metadata": table_data.get("metadata", {})
            }
        )
        
        # Generate description
        description = self._generate_description(table_data)
        
        return {
            "element_type": "table",
            "page_number": page_number,
            "file_path": file_path,  # Image path
            "table_json_path": table_json_path,  # JSON path
            "table_markdown": markdown,
            "text_annotation": description,
            "metadata": table_data.get("metadata", {})
        }
    
    def _extract_table_image(
        self,
        pdf_path: str,
        bbox: dict,
        page_number: int,
        document_id: str,
        element_id: str
    ) -> str:
        """
        Extract table region from PDF as an image.
        
        Args:
            pdf_path: Path to PDF file
            bbox: Bounding box with l, t, r, b, coord_origin
            page_number: Page number (1-indexed)
            document_id: Document ID
            element_id: Element ID
            
        Returns:
            Storage path to saved image
        """
        import fitz  # PyMuPDF
        
        # Open PDF
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[page_number - 1]  # 0-indexed
        page_height = page.rect.height
        
        # Convert coordinates based on origin
        # Docling uses BOTTOMLEFT, PyMuPDF uses TOPLEFT
        coord_origin = str(bbox.get('coord_origin', 'BOTTOMLEFT'))
        if 'BOTTOMLEFT' in coord_origin:
            y0 = page_height - bbox['t']  # top in TOPLEFT
            y1 = page_height - bbox['b']  # bottom in TOPLEFT
        else:
            y0 = bbox['t']
            y1 = bbox['b']
        
        rect = fitz.Rect(bbox['l'], y0, bbox['r'], y1)
        
        # Extract as image with 2x scaling for better quality
        pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        pdf_doc.close()
        
        # Save to storage
        file_path = self.storage.save_image(
            document_id=document_id,
            element_id=element_id,
            image_data=img_bytes,
            extension="png"
        )
        
        return file_path
    
    def _generate_description(self, table_data: dict) -> str:
        """
        Generate text description of table.
        
        Uses markdown content to create description.
        """
        markdown = table_data.get("markdown", table_data.get("text", ""))
        
        if not markdown or markdown == "| Table data unavailable |":
            return "Table with 0 rows"
        
        # Count rows from markdown
        lines = [l.strip() for l in markdown.split('\n') if l.strip() and l.strip().startswith('|')]
        row_count = len([l for l in lines if not all(c in '|-: ' for c in l)])
        
        description = f"Table with {row_count} rows"
        
        # Try to extract first line as headers
        if lines:
            first_line = lines[0]
            headers = [h.strip() for h in first_line.split('|') if h.strip()][:3]
            if headers:
                description += f". Headers: {', '.join(headers)}"
        
        return description
    
    def generate_llm_description(
        self,
        markdown: str,
        llm_service: Any
    ) -> str:
        """
        Generate LLM-based description of table.
        
        Args:
            markdown: Table markdown
            llm_service: LLM service with generate capabilities
            
        Returns:
            Generated description
        """
        prompt = f"""Describe this table in 1-2 sentences. Focus on what data it contains and key insights.

Table:
{markdown[:500]}"""  # Limit to first 500 chars
        
        try:
            description = llm_service.generate(
                prompt=prompt,
                max_tokens=100
            )
            return description.strip()
        except Exception as e:
            logger.warning(f"Failed to generate LLM description: {e}")
            return self._generate_description({"markdown": markdown})
