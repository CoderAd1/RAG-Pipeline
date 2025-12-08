"""Text chunking strategies for RAG pipelines."""

from typing import List, Dict, Any
from loguru import logger


class FixedSizeChunker:
    """Fixed-size text chunking for basic RAG."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, page_number: int = 1) -> List[Dict[str, Any]]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to chunk
            page_number: Page number for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings
                last_period = chunk_text.rfind('. ')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:  # Only break if we're past halfway
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append({
                "text": chunk_text.strip(),
                "page_number": page_number,
                "chunk_index": chunk_index,
                "metadata": {
                    "chunk_size": len(chunk_text),
                    "start_char": start,
                    "end_char": end
                }
            })
            
            chunk_index += 1
            start = end - self.overlap
        
        logger.debug(f"Created {len(chunks)} fixed-size chunks from text (page {page_number})")
        return chunks


class SemanticChunker:
    """Semantic chunking for advanced RAG with structure awareness."""
    
    def __init__(self, min_chunk_size: int = 500, max_chunk_size: int = 1500):
        """
        Initialize semantic chunker.
        
        Args:
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_document(
        self,
        sections: List[Dict[str, Any]],
        preserve_structure: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk document respecting semantic structure.
        
        Args:
            sections: List of document sections with hierarchy
            preserve_structure: Whether to preserve section boundaries
            
        Returns:
            List of semantic chunks with metadata
        """
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_chunks = self._chunk_section(section, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        logger.debug(f"Created {len(chunks)} semantic chunks from document")
        return chunks
    
    def _chunk_section(
        self,
        section: Dict[str, Any],
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Chunk a single section."""
        text = section.get("text", "")
        page_number = section.get("page", 1)
        section_type = section.get("type", "paragraph")
        
        if not text or len(text) < self.min_chunk_size:
            # Small section, keep as single chunk
            return [{
                "text": text,
                "page_number": page_number,
                "chunk_index": start_index,
                "chunk_type": "text",
                "metadata": {
                    "section_type": section_type,
                    "visual_refs": section.get("visual_refs", [])
                }
            }]
        
        # Split large sections into paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_idx = start_index
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page_number": page_number,
                        "chunk_index": chunk_idx,
                        "chunk_type": "text",
                        "metadata": {
                            "section_type": section_type,
                            "visual_refs": section.get("visual_refs", [])
                        }
                    })
                    chunk_idx += 1
                current_chunk = para + "\n\n"
        
        # Add remaining text
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "page_number": page_number,
                "chunk_index": chunk_idx,
                "chunk_type": "text",
                "metadata": {
                    "section_type": section_type,
                    "visual_refs": section.get("visual_refs", [])
                }
            })
        
        return chunks
    
    def create_visual_context_chunk(
        self,
        visual_element: Dict[str, Any],
        context_text: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """
        Create a chunk for visual element context.
        
        Args:
            visual_element: Visual element data
            context_text: Surrounding context text
            chunk_index: Chunk index
            
        Returns:
            Chunk dictionary
        """
        element_type = visual_element.get("element_type", "unknown")
        description = visual_element.get("text_annotation", "")
        
        combined_text = f"{context_text}\n\n[{element_type.upper()}]: {description}"
        
        return {
            "text": combined_text,
            "page_number": visual_element.get("page_number", 1),
            "chunk_index": chunk_index,
            "chunk_type": f"{element_type}_context",
            "metadata": {
                "visual_element_id": visual_element.get("id"),
                "element_type": element_type
            }
        }
