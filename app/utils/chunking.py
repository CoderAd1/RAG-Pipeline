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

    def __init__(
        self,
        min_chunk_size: int = 500,
        max_chunk_size: int = 1500,
        table_context_window: int = 500
    ):
        """
        Initialize semantic chunker.

        Args:
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            table_context_window: Characters of context to include around tables
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.table_context_window = table_context_window

    def chunk_document(
        self,
        sections: List[Dict[str, Any]],
        tables: List[Dict[str, Any]] = None,
        preserve_structure: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk document respecting semantic structure with table enrichment.

        Args:
            sections: List of document sections with hierarchy
            tables: List of table data to enrich chunks with
            preserve_structure: Whether to preserve section boundaries

        Returns:
            List of semantic chunks with metadata, including table-enriched chunks
        """
        chunks = []
        chunk_index = 0

        # Create table-to-page mapping for quick lookup
        tables = tables or []
        tables_by_page = {}
        for table in tables:
            page = table.get("page", 1)
            if page not in tables_by_page:
                tables_by_page[page] = []
            tables_by_page[page].append(table)

        # Group sections by page for context-aware chunking
        sections_by_page = {}
        for section in sections:
            page = section.get("page", 1)
            if page not in sections_by_page:
                sections_by_page[page] = []
            sections_by_page[page].append(section)

        # Process each page
        all_pages = sorted(set(list(sections_by_page.keys()) + list(tables_by_page.keys())))

        for page in all_pages:
            page_sections = sections_by_page.get(page, [])
            page_tables = tables_by_page.get(page, [])

            # Chunk page sections
            for section in page_sections:
                section_chunks = self._chunk_section(section, chunk_index)
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)

            # Create dedicated table chunks with context
            for table in page_tables:
                table_chunk = self._create_table_chunk(
                    table=table,
                    page_sections=page_sections,
                    chunk_index=chunk_index
                )
                chunks.append(table_chunk)
                chunk_index += 1

        table_count = len([c for c in chunks if 'table' in c.get('chunk_type', '')])
        logger.debug(
            f"Created {len(chunks)} semantic chunks from document "
            f"({table_count} table chunks)"
        )
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

    def _create_table_chunk(
        self,
        table: Dict[str, Any],
        page_sections: List[Dict[str, Any]],
        chunk_index: int
    ) -> Dict[str, Any]:
        """Create a highly searchable table-enriched chunk with extensive metadata."""
        table_markdown = table.get("markdown", table.get("text", ""))
        page_num = table.get("page", 1)

        # Parse table for searchable metadata
        table_metadata = self._extract_table_metadata(table_markdown)

        # Get surrounding context (text before/after table on same page)
        context_parts = []
        page_text = "\n\n".join([s.get("text", "") for s in page_sections if s.get("text", "").strip()])

        # Limit context to table_context_window
        if page_text:
            if len(page_text) > self.table_context_window:
                # Take first part of page text as context
                context_parts.append(page_text[:self.table_context_window] + "...")
            else:
                context_parts.append(page_text)

        # Build the enriched chunk text with extensive searchable metadata
        chunk_text_parts = []

        # Add searchable keywords at the top for better embedding
        keywords_parts = []
        if table_metadata["headers"]:
            keywords_parts.append(f"Table columns: {', '.join(table_metadata['headers'])}")
        if table_metadata["row_labels"]:
            keywords_parts.append(f"Variables/rows: {', '.join(table_metadata['row_labels'])}")
        if table_metadata["statistical_terms"]:
            keywords_parts.append(f"Statistical measures: {', '.join(table_metadata['statistical_terms'])}")

        if keywords_parts:
            chunk_text_parts.append("TABLE METADATA:\n" + "\n".join(keywords_parts))

        # ULTRA-SEARCHABLE: Extract all table cell values as plain text
        plain_text_data = self._extract_table_plain_text(table_markdown, table_metadata)
        if plain_text_data:
            chunk_text_parts.append(f"\nTABLE CONTENT (SEARCHABLE):\n{plain_text_data}")

        if context_parts:
            chunk_text_parts.append("\nCONTEXT:\n" + "\n".join(context_parts))

        # Add table with clear markers for better retrieval
        chunk_text_parts.append(f"\nTABLE DATA (Page {page_num}):\n{table_markdown}")

        # Add interpretive description if available
        table_annotation = table.get("text_annotation", "")
        if table_annotation:
            chunk_text_parts.append(f"\nTABLE DESCRIPTION: {table_annotation}")

        combined_text = "\n".join(chunk_text_parts)

        return {
            "text": combined_text,
            "page_number": page_num,
            "chunk_index": chunk_index,
            "chunk_type": "table_enriched",
            "metadata": {
                "section_type": "table",
                "has_table": True,
                "table_markdown": table_markdown,
                "bbox": table.get("bbox"),
                "table_headers": table_metadata["headers"],
                "table_row_labels": table_metadata["row_labels"],
                "statistical_terms": table_metadata["statistical_terms"]
            }
        }

    def _extract_table_plain_text(
        self,
        markdown: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Extract ALL table data as plain searchable text.

        Creates natural language sentences for every row like:
        "variable_name: header1=value1, header2=value2, header3=value3"

        This makes the table ULTRA-SEARCHABLE by embedding model.
        """
        if not markdown or markdown.strip() == "| Table data unavailable |":
            return ""

        # Parse table structure
        lines = [l.strip() for l in markdown.split('\n') if l.strip() and l.strip().startswith('|')]
        data_lines = [l for l in lines if not all(c in '|-: ' for c in l)]

        if len(data_lines) < 2:  # Need at least header + 1 data row
            return ""

        # Parse headers
        header_line = data_lines[0]
        headers = [h.strip() for h in header_line.split('|') if h.strip()]

        if not headers:
            return ""

        # Parse data rows and create natural language descriptions
        plain_text_parts = []

        for row_line in data_lines[1:]:
            cells = [c.strip() for c in row_line.split('|') if c.strip()]

            if not cells or len(cells) != len(headers):
                continue

            # Create sentence for this row
            row_label = cells[0]  # First column is usually the row label/variable

            # Build natural language: "variable_name: header1=value1, header2=value2, ..."
            value_pairs = []
            for i, (header, value) in enumerate(zip(headers, cells)):
                if i == 0:
                    continue  # Skip first column (row label)
                if value and value not in ['-', 'NA', 'N/A', '']:
                    value_pairs.append(f"{header} = {value}")

            if value_pairs:
                row_sentence = f"{row_label}: {', '.join(value_pairs)}"
                plain_text_parts.append(row_sentence)

        return "\n".join(plain_text_parts)

    def _extract_table_metadata(self, markdown: str) -> Dict[str, Any]:
        """
        Extract searchable metadata from table markdown.

        Returns:
            Dict with headers, row_labels, and statistical_terms
        """
        if not markdown or markdown.strip() == "| Table data unavailable |":
            return {"headers": [], "row_labels": [], "statistical_terms": []}

        # Parse table structure
        lines = [l.strip() for l in markdown.split('\n') if l.strip() and l.strip().startswith('|')]

        # Remove separator lines
        data_lines = [l for l in lines if not all(c in '|-: ' for c in l)]

        if not data_lines:
            return {"headers": [], "row_labels": [], "statistical_terms": []}

        # Extract headers (first line)
        headers = []
        if data_lines:
            first_line = data_lines[0]
            headers = [h.strip() for h in first_line.split('|') if h.strip()]

        # Extract row labels (first column of each data row, excluding header)
        row_labels = []
        if len(data_lines) > 1:
            for line in data_lines[1:]:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    # Clean up row labels - handle various formats
                    label = cells[0]
                    # Common patterns in statistical tables
                    if label and label not in ['(Intercept)', '---', '...']:
                        row_labels.append(label)

        # Identify statistical keywords in the table
        stat_keywords = ['p-value', 'p.value', 'pvalue', 'p value', 'coefficient', 'coef',
                        'std', 'stderr', 'se', 'conf', 'interval', 't-stat', 't value',
                        'z-score', 'chi-square', 'df', 'mean', 'median', 'correlation',
                        'r-squared', 'r2', 'beta', 'odds', 'ratio', 'significance',
                        'alpha', 'estimate', 'log', 'ln', 'exp', 'population',
                        'regression', 'model', 'intercept', 'residual']

        statistical_terms = []
        markdown_lower = markdown.lower()
        for term in stat_keywords:
            if term in markdown_lower:
                if term not in statistical_terms:
                    statistical_terms.append(term)

        return {
            "headers": headers[:10],  # First 10 headers
            "row_labels": row_labels[:15],  # First 15 row labels
            "statistical_terms": statistical_terms
        }

    def create_visual_context_chunk(
        self,
        visual_element: Dict[str, Any],
        context_text: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """Create a chunk for visual element context."""
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
