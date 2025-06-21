from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from typing import Optional
from database import SessionLocal, Document, DocumentSchema, Entity, EntitySchema
from file_utils import find_text_file_paths, read_text_file
from text_utils import format_document_display
from pathlib import Path

router = APIRouter(prefix="/api", tags=["documents"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/documents/", response_model=list[DocumentSchema])
async def get_documents(subject: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Get all documents, optionally filtered by subject.
    Returns documents sorted by upload_date (newest first).
    """
    query = db.query(Document)
    
    # Filter by subject if provided
    if subject:
        query = query.filter(Document.subject == subject)
    
    # Sort by upload_date (newest first)
    documents = query.order_by(Document.upload_date.desc()).all()
    return documents

@router.get("/documents/subjects/")
async def get_unique_subjects(db: Session = Depends(get_db)):
    """Endpoint to get all unique subjects from documents."""
    try:
        subjects = db.query(Document.subject).filter(Document.subject.isnot(None)).distinct().all()
        unique_subjects = [subject[0] for subject in subjects if subject[0]]
        return {"subjects": unique_subjects}
    except Exception as e:
        print(f"ERROR: Failed to get unique subjects: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve unique subjects")

@router.get("/documents/{document_id}/entities", response_model=list[EntitySchema])
async def get_document_entities(document_id: int, db: Session = Depends(get_db)):
    """Endpoint to retrieve extracted entities for a specific document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    entities = db.query(Entity).filter(Entity.document_id == document_id).all()
    return entities

@router.get("/documents/{document_id}/text")
async def get_document_text(document_id: int, db: Session = Depends(get_db)):
    """Endpoint to retrieve the processed text content of a specific document with enhanced formatting."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found in database")
    
    if not document.path_to_text or not Path(document.path_to_text).exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Text file not found on server for document ID {document_id}. Path: {document.path_to_text}"
        )
    
    # Read the raw text content
    raw_text = read_text_file(Path(document.path_to_text))
    
    # Format it beautifully for display
    formatted_text = format_document_display(
        text=raw_text,
        document_title=document.filename,
        document_id=document_id
    )
    
    return PlainTextResponse(content=formatted_text, media_type="text/plain; charset=utf-8")

@router.get("/documents/{document_id}/raw-text")
async def get_document_raw_text(document_id: int, formatted: bool = False, db: Session = Depends(get_db)):
    """
    Endpoint to retrieve the completely raw, unprocessed text content of a specific document.
    Falls back to processed text if raw text file doesn't exist.
    
    Args:
        document_id: The ID of the document
        formatted: Whether to apply enhanced formatting (default: False for backward compatibility)
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found in database")
    
    if not document.path_to_text:
        raise HTTPException(status_code=404, detail=f"No text file path found for document ID {document_id}")
    
    print(f"DEBUG: Stored path in database: {document.path_to_text}")
    
    raw_paths_to_try, processed_paths_to_try = find_text_file_paths(document.path_to_text)
    
    # Try raw files first
    for raw_path in raw_paths_to_try:
        if raw_path.exists():
            print(f"DEBUG: Found raw text file: {raw_path}")
            try:
                raw_text = read_text_file(raw_path)
                
                if formatted:
                    # Apply enhanced formatting
                    formatted_text = format_document_display(
                        text=raw_text,
                        document_title=document.filename,
                        document_id=document_id
                    )
                    return PlainTextResponse(content=formatted_text, media_type="text/plain; charset=utf-8")
                else:
                    # Return raw text as-is
                    return PlainTextResponse(content=raw_text, media_type="text/plain; charset=utf-8")
                    
            except Exception as e:
                print(f"ERROR: Failed to read raw file {raw_path}: {e}")
                continue
    
    # Fallback to processed files
    print(f"DEBUG: No raw text files found, trying processed files...")
    for processed_path_option in processed_paths_to_try:
        if processed_path_option.exists():
            print(f"DEBUG: Found processed text file: {processed_path_option}")
            try:
                content = read_text_file(processed_path_option)
                print(f"DEBUG: Successfully read {len(content)} characters from processed file")
                
                if formatted:
                    # Apply enhanced formatting
                    formatted_text = format_document_display(
                        text=content,
                        document_title=document.filename,
                        document_id=document_id
                    )
                    return PlainTextResponse(content=formatted_text, media_type="text/plain; charset=utf-8")
                else:
                    # Return processed text as-is
                    return PlainTextResponse(content=content, media_type="text/plain; charset=utf-8")
                    
            except Exception as e:
                print(f"ERROR: Failed to read processed file {processed_path_option}: {e}")
                continue
    
    # If we get here, no files were found
    all_tried_paths = raw_paths_to_try + processed_paths_to_try
    raise HTTPException(
        status_code=404, 
        detail=f"No text files found for document ID {document_id}. Tried paths: {[str(p) for p in all_tried_paths]}"
    )

@router.get("/documents/{document_id}/formatted-text")
async def get_document_formatted_text(document_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to retrieve document text with enhanced formatting for better readability.
    This endpoint always returns beautifully formatted text with headers, sections, and statistics.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found in database")
    
    if not document.path_to_text:
        raise HTTPException(status_code=404, detail=f"No text file path found for document ID {document_id}")
    
    # Try to find and read text files (raw first, then processed)
    raw_paths_to_try, processed_paths_to_try = find_text_file_paths(document.path_to_text)
    
    text_content = None
    
    # Try raw files first
    for raw_path in raw_paths_to_try:
        if raw_path.exists():
            try:
                text_content = read_text_file(raw_path)
                break
            except Exception as e:
                print(f"ERROR: Failed to read raw file {raw_path}: {e}")
                continue
    
    # Fallback to processed files
    if text_content is None:
        for processed_path_option in processed_paths_to_try:
            if processed_path_option.exists():
                try:
                    text_content = read_text_file(processed_path_option)
                    break
                except Exception as e:
                    print(f"ERROR: Failed to read processed file {processed_path_option}: {e}")
                    continue
    
    # If no text content found
    if text_content is None:
        all_tried_paths = raw_paths_to_try + processed_paths_to_try
        raise HTTPException(
            status_code=404, 
            detail=f"No text files found for document ID {document_id}. Tried paths: {[str(p) for p in all_tried_paths]}"
        )
    
    # Apply enhanced formatting
    formatted_text = format_document_display(
        text=text_content,
        document_title=document.filename,
        document_id=document_id
    )
    
    return PlainTextResponse(content=formatted_text, media_type="text/plain; charset=utf-8")
