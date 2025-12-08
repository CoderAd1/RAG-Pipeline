"""Advanced RAG API endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
import tempfile
import time
from pathlib import Path
from loguru import logger
from supabase import Client
import uuid

from app.models.schemas import (
    AdvancedUploadResponse,
    QueryRequest,
    AdvancedQueryResponse,
    SourceReference,
    VisualReference
)
from app.core.database import get_supabase
from app.api.dependencies import (
    get_embedding_service,
    get_llm_service,
    get_qdrant_advanced,
    get_advanced_pdf_processor,
    get_table_processor,
    get_image_processor
)
from app.utils.validators import FileValidator
from app.utils.chunking import SemanticChunker
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.llm.llm_service import LLMService
from app.services.vector_store.qdrant_advanced import QdrantAdvancedService
from app.services.pdf.advanced_processor import AdvancedPDFProcessor
from app.services.visual.table_processor import TableProcessor
from app.services.visual.image_processor import ImageProcessor

router = APIRouter(prefix="/advanced", tags=["Advanced RAG"])


@router.post("/upload", response_model=AdvancedUploadResponse)
async def upload_pdf_advanced(
    file: UploadFile = File(...),
    supabase: Client = Depends(get_supabase),
    pdf_processor: AdvancedPDFProcessor = Depends(get_advanced_pdf_processor),
    table_processor: TableProcessor = Depends(get_table_processor),
    image_processor: ImageProcessor = Depends(get_image_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    qdrant_service: QdrantAdvancedService = Depends(get_qdrant_advanced),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Upload and process PDF for advanced RAG with multimodal support.
    
    - Extracts text, tables, and images using Docling
    - Creates semantic chunks
    - Processes visual elements with LLM-generated descriptions
    - Stores in advanced collections
    """
    try:
        # Validate file
        sanitized_filename = FileValidator.validate_and_sanitize(file)
        logger.info(f"Processing advanced upload: {sanitized_filename}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Extract document with structure
            logger.info("Extracting document with Docling...")
            extracted_data = pdf_processor.extract_document(tmp_path)
            
            # Create document record
            doc_data = {
                "filename": sanitized_filename,
                "file_path": None,
                "total_pages": extracted_data["total_pages"],
                "processing_status": "processing",
                "ingestion_type": "advanced",
                "metadata": extracted_data["metadata"]
            }
            
            result = supabase.table("documents").insert(doc_data).execute()
            document_id = result.data[0]["id"]
            logger.info(f"Created document record: {document_id}")
            
            # Process visual elements
            visual_elements_data = []
            tables_count = 0
            images_count = 0
            
            # Process tables with image extraction
            for table_data in extracted_data.get("tables", []):
                element_id = str(uuid.uuid4())
                processed_table = table_processor.process_table(
                    table_data=table_data,
                    document_id=document_id,
                    element_id=element_id,
                    page_number=table_data.get("page", 1),
                    pdf_path=str(tmp_path)  # Pass PDF path for image extraction
                )
                
                # Table descriptions disabled - using simple descriptions
                # to avoid API quota issues. Uncomment below to enable LLM table descriptions.
                # try:
                #     if processed_table.get("table_markdown"):
                #         llm_description = table_processor.generate_llm_description(
                #             markdown=processed_table["table_markdown"],
                #             llm_service=llm_service
                #         )
                #         processed_table["text_annotation"] = llm_description
                #         logger.debug(f"Generated LLM description for table on page {processed_table['page_number']}")
                # except Exception as e:
                #     logger.warning(f"Failed to generate LLM description for table: {e}")
                
                visual_element = {
                    "id": element_id,
                    "document_id": document_id,
                    "element_type": "table",
                    "page_number": processed_table["page_number"],
                    "file_path": processed_table.get("file_path"),  # Table image path
                    "table_markdown": processed_table.get("table_markdown"),
                    "text_annotation": processed_table["text_annotation"],
                    "ingestion_type": "advanced",
                    "metadata": processed_table.get("metadata", {})
                }
                visual_elements_data.append(visual_element)
                tables_count += 1
            
            # Process images with LLM descriptions
            for image_data in extracted_data.get("images", []):
                element_id = str(uuid.uuid4())
                processed_image = image_processor.process_image(
                    image_data=image_data,
                    document_id=document_id,
                    element_id=element_id,
                    page_number=image_data.get("page", 1)
                )
                
                # Vision descriptions disabled - using caption-based descriptions
                # to avoid API quota issues. Uncomment below to enable LLM vision descriptions.
                # try:
                #     if processed_image.get("file_path"):
                #         from app.utils.file_storage import storage
                #         absolute_path = storage.get_file_path(processed_image["file_path"])
                #         
                #         vision_description = image_processor.generate_vision_description(
                #             image_path=str(absolute_path),
                #             llm_service=llm_service
                #         )
                #         processed_image["text_annotation"] = vision_description
                #         logger.info(f"Generated vision description for image on page {processed_image['page_number']}")
                # except Exception as e:
                #     logger.warning(f"Failed to generate vision description for image: {e}")
                
                visual_element = {
                    "id": element_id,
                    "document_id": document_id,
                    "element_type": processed_image["element_type"],
                    "page_number": processed_image["page_number"],
                    "bounding_box": processed_image.get("bounding_box"),
                    "file_path": processed_image.get("file_path"),
                    "text_annotation": processed_image["text_annotation"],
                    "ingestion_type": "advanced",
                    "metadata": processed_image.get("metadata", {})
                }
                visual_elements_data.append(visual_element)
                images_count += 1
            
            # Store visual elements in database
            if visual_elements_data:
                supabase.table("visual_elements").insert(visual_elements_data).execute()
                logger.info(f"Stored {len(visual_elements_data)} visual elements")
            
            # Create semantic chunks
            logger.info("Creating semantic chunks...")
            chunker = SemanticChunker(min_chunk_size=500, max_chunk_size=1500)
            chunks = chunker.chunk_document(
                sections=extracted_data.get("sections", []),
                preserve_structure=True
            )
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Store chunks in database
            chunk_records = []
            for chunk in chunks:
                # Link to visual elements if referenced
                visual_refs = chunk.get("metadata", {}).get("visual_refs", [])
                
                chunk_record = {
                    "document_id": document_id,
                    "chunk_text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "page_number": chunk["page_number"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "ingestion_type": "advanced",
                    "visual_element_ids": visual_refs,
                    "metadata": chunk.get("metadata", {})
                }
                chunk_records.append(chunk_record)
            
            chunks_result = supabase.table("chunks").insert(chunk_records).execute()
            stored_chunks = chunks_result.data
            logger.info(f"Stored {len(stored_chunks)} chunks")
            
            # Generate embeddings for text chunks
            logger.info("Generating text embeddings...")
            texts = [chunk["text"] for chunk in chunks]
            text_embeddings = embedding_service.embed_batch(texts)
            
            # Prepare chunks with IDs
            chunks_with_ids = []
            for chunk, stored_chunk in zip(chunks, stored_chunks):
                chunk_copy = chunk.copy()
                chunk_copy["chunk_id"] = stored_chunk["id"]
                chunk_copy["visual_element_ids"] = stored_chunk.get("visual_element_ids", [])
                chunks_with_ids.append(chunk_copy)
            
            # Store text chunks in Qdrant
            logger.info("Storing text vectors...")
            text_vector_ids = qdrant_service.insert_text_chunks(
                chunks=chunks_with_ids,
                embeddings=text_embeddings,
                document_id=document_id
            )
            
            # Store text embedding records
            text_embedding_records = []
            for chunk_id, vector_id in zip([c["id"] for c in stored_chunks], text_vector_ids):
                text_embedding_records.append({
                    "chunk_id": chunk_id,
                    "collection_name": "advanced_text_collection",
                    "vector_id": vector_id,
                    "embedding_model": embedding_service.get_model_name(),
                    "ingestion_type": "advanced"
                })
            
            supabase.table("embeddings").insert(text_embedding_records).execute()
            
            # Generate embeddings for visual elements
            if visual_elements_data:
                logger.info("Generating visual embeddings...")
                visual_texts = [ve["text_annotation"] for ve in visual_elements_data]
                visual_embeddings = embedding_service.embed_batch(visual_texts)
                
                # Prepare visual elements with IDs
                visual_with_ids = []
                for ve in visual_elements_data:
                    visual_with_ids.append({
                        "element_id": ve["id"],
                        "element_type": ve["element_type"],
                        "text_annotation": ve["text_annotation"],
                        "page_number": ve["page_number"],
                        "file_path": ve.get("file_path", ""),
                        "metadata": ve.get("metadata", {})
                    })
                
                # Store visual vectors
                logger.info("Storing visual vectors...")
                visual_vector_ids = qdrant_service.insert_visual_elements(
                    visual_elements=visual_with_ids,
                    embeddings=visual_embeddings,
                    document_id=document_id
                )
                
                # Store visual embedding records
                visual_embedding_records = []
                for ve_id, vector_id in zip([ve["id"] for ve in visual_elements_data], visual_vector_ids):
                    visual_embedding_records.append({
                        "visual_element_id": ve_id,
                        "collection_name": "advanced_visual_collection",
                        "vector_id": vector_id,
                        "embedding_model": embedding_service.get_model_name(),
                        "ingestion_type": "advanced"
                    })
                
                supabase.table("embeddings").insert(visual_embedding_records).execute()
            
            # Update document status
            supabase.table("documents").update({
                "processing_status": "completed"
            }).eq("id", document_id).execute()
            
            logger.success(f"Successfully processed advanced document {document_id}")
            
            return AdvancedUploadResponse(
                document_id=document_id,
                filename=sanitized_filename,
                total_pages=extracted_data["total_pages"],
                chunks_created=len(stored_chunks),
                visual_elements_count=len(visual_elements_data),
                tables_extracted=tables_count,
                images_extracted=images_count,
                ingestion_type="advanced",
                processing_status="completed",
                message="Document processed successfully with multimodal extraction"
            )
            
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    except Exception as e:
        logger.error(f"Advanced upload failed: {e}")
        
        # Update document status if created
        if 'document_id' in locals():
            try:
                supabase.table("documents").update({
                    "processing_status": "failed",
                    "error_message": str(e)
                }).eq("id", document_id).execute()
            except:
                pass
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=AdvancedQueryResponse)
async def query_advanced(
    request: QueryRequest,
    supabase: Client = Depends(get_supabase),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    qdrant_service: QdrantAdvancedService = Depends(get_qdrant_advanced),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Query the advanced RAG system with multimodal support.
    
    - Searches both text and visual collections
    - Retrieves tables and images
    - Generates comprehensive answer
    """
    try:
        start_time = time.time()
        
        logger.info(f"Processing advanced query: {request.query[:100]}...")
        
        # Embed query
        query_embedding = embedding_service.embed_text(request.query)
        
        # Search both collections
        top_k = request.top_k or 10
        text_k = int(top_k * 0.7)  # 70% from text
        visual_k = int(top_k * 0.3)  # 30% from visual
        
        # Search text collection
        text_results = qdrant_service.search_text(
            query_embedding=query_embedding,
            top_k=text_k
        )
        
        # Search visual collection
        visual_results = qdrant_service.search_visual(
            query_embedding=query_embedding,
            top_k=visual_k
        )
        
        if not text_results and not visual_results:
            return AdvancedQueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                visual_elements=[],
                ingestion_type="advanced",
                query_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Get document names
        all_doc_ids = list(set(
            [r["document_id"] for r in text_results] +
            [r["document_id"] for r in visual_results]
        ))
        docs_result = supabase.table("documents").select("id, filename").in_("id", all_doc_ids).execute()
        doc_map = {doc["id"]: doc["filename"] for doc in docs_result.data}
        
        # Get visual element details
        visual_element_ids = [r["element_id"] for r in visual_results]
        visual_details = {}
        if visual_element_ids:
            ve_result = supabase.table("visual_elements").select("*").in_("id", visual_element_ids).execute()
            visual_details = {ve["id"]: ve for ve in ve_result.data}
        
        # Prepare context for LLM
        context_chunks = []
        for result in text_results:
            context_chunks.append({
                "text": result["text"],
                "page": result["page"],
                "document_name": doc_map.get(result["document_id"], "Unknown")
            })
        
        # Prepare visual elements for LLM
        visual_elements_for_llm = []
        for result in visual_results:
            ve_detail = visual_details.get(result["element_id"], {})
            visual_elements_for_llm.append({
                "element_type": result["element_type"],
                "description": result["text_annotation"],
                "page_number": result["page"],
                "table_markdown": ve_detail.get("table_markdown"),
                "document_name": doc_map.get(result["document_id"], "Unknown")
            })
        
        # Generate answer
        logger.info("Generating multimodal answer...")
        answer = llm_service.generate_answer(
            question=request.query,
            context_chunks=context_chunks,
            visual_elements=visual_elements_for_llm
        )
        
        # Prepare source references
        sources = []
        for result in text_results:
            sources.append(SourceReference(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                document_name=doc_map.get(result["document_id"], "Unknown"),
                page_number=result["page"],
                text_snippet=result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                relevance_score=result["score"],
                chunk_type=result.get("chunk_type")
            ))
        
        # Prepare visual references
        visual_refs = []
        for result in visual_results:
            ve_detail = visual_details.get(result["element_id"], {})
            visual_refs.append(VisualReference(
                element_id=result["element_id"],
                element_type=result["element_type"],
                document_id=result["document_id"],
                page_number=result["page"],
                description=result["text_annotation"],
                file_path=ve_detail.get("file_path"),
                table_markdown=ve_detail.get("table_markdown"),
                relevance_score=result["score"]
            ))
        
        query_time_ms = int((time.time() - start_time) * 1000)
        
        # Log query
        try:
            supabase.table("query_logs").insert({
                "query_text": request.query,
                "ingestion_type": "advanced",
                "collection_searched": ["advanced_text_collection", "advanced_visual_collection"],
                "retrieved_chunk_ids": [r["chunk_id"] for r in text_results],
                "response_time_ms": query_time_ms
            }).execute()
        except Exception as log_error:
            logger.warning(f"Failed to log query: {log_error}")
        
        logger.success(f"Advanced query completed in {query_time_ms}ms")
        
        return AdvancedQueryResponse(
            answer=answer,
            sources=sources,
            visual_elements=visual_refs,
            ingestion_type="advanced",
            query_time_ms=query_time_ms
        )
    
    except Exception as e:
        logger.error(f"Advanced query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
