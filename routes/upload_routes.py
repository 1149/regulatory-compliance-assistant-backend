# File upload and management routes
import os
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pdfminer.high_level import extract_text as pdfminer_extract_text

from database import SessionLocal, Document, Entity
from file_utils import setup_upload_directory, get_file_paths, determine_pdf_path_from_text_path, delete_file_safely
from text_utils import clean_extracted_text
from nlp_utils import extract_entities_with_spacy, post_process_spacy_entities
from ai_services import identify_compliance_clauses, generate_document_embeddings

router = APIRouter(prefix="/api", tags=["upload"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload-document/")
async def upload_document(
    file: UploadFile = File(...),
    subject: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload and process a PDF document."""
    # Validate PDF file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed."
        )
    
    upload_dir = setup_upload_directory()
    pdf_path, text_path, raw_text_path = get_file_paths(file.filename, upload_dir)

    # Save uploaded PDF file
    try:
        with open(pdf_path, "wb+") as file_object:
            content = await file.read()
            file_object.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save PDF file: {e}"
        )

    # Extract and process text
    try:
        # Extract raw text using pdfminer
        raw_extracted_text = pdfminer_extract_text(str(pdf_path))
        
        # Save raw text
        with open(raw_text_path, "w", encoding="utf-8") as raw_text_file:
            raw_text_file.write(raw_extracted_text)

        # Clean text for processing
        extracted_text = clean_extracted_text(raw_extracted_text)
        
        print(f"DEBUG: Final Cleaned Extracted Text (first 500 chars):\n{extracted_text[:500]}...\n---End Final Cleaned Text---")
        
        # Save processed text
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)

    except Exception as e:
        # Cleanup files on error
        for path in [pdf_path, raw_text_path]:
            if path.exists():
                os.remove(path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract text from PDF: {e}"
        )

    # Identify compliance clauses
    identified_clauses_data = identify_compliance_clauses(extracted_text, file.filename)

    # Extract entities using SpaCy
    extracted_entities_data_spacy = []
    try:
        if extracted_text:
            extracted_entities_data_spacy = extract_entities_with_spacy(extracted_text)
            extracted_entities_data_spacy = post_process_spacy_entities(extracted_entities_data_spacy)
        
        print(f"DEBUG: SpaCy Extracted Entities (Post-Processed):\n{extracted_entities_data_spacy}\n---End SpaCy Entities---")
    except Exception as e:
        print(f"Warning: Failed to extract SpaCy entities for {file.filename}: {e}")

    # Generate embeddings
    embeddings_json_string = generate_document_embeddings(extracted_text, file.filename)

    # Save to database
    try:
        db_document = Document(
            filename=file.filename,
            upload_date=datetime.utcnow(),
            status="processed_text",
            path_to_text=str(text_path),
            embeddings=embeddings_json_string,
            subject=subject
        )
        db.add(db_document)

        # Save all entities (SpaCy + clauses)
        all_entities_to_save = extracted_entities_data_spacy + identified_clauses_data

        for entity_data in all_entities_to_save:
            print("Saving entity:", entity_data)
            db_entity = Entity(
                document=db_document,
                text=entity_data["text"],
                label=entity_data["label"],
                start_char=entity_data["start_char"],
                end_char=entity_data["end_char"]
            )
            db.add(db_entity)

        db.commit()
        db.refresh(db_document)

        return {
            "filename": db_document.filename,
            "text_location": db_document.path_to_text,
            "id": db_document.id,
            "embeddings_saved": bool(embeddings_json_string),
            "entities_identified": len(all_entities_to_save),
            "message": "PDF uploaded, text, entities & clauses extracted, metadata saved successfully!"
        }
        
    except Exception as e:
        db.rollback()
        # Cleanup files on database error
        for path in [pdf_path, text_path, raw_text_path]:
            if path.exists():
                os.remove(path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save to database: {e}"
        )

@router.delete("/documents/{document_id}/")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Permanently delete a document and all associated data.
    This includes database records, entities, and files.
    """
    try:
        print(f"DEBUG: Starting deletion process for document ID: {document_id}")
        
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Determine file paths
        text_path = None
        pdf_path = None
        
        if document.path_to_text:
            text_path = Path(document.path_to_text)
            pdf_path = determine_pdf_path_from_text_path(text_path)
        
        print(f"DEBUG: Text file path: {text_path}")
        print(f"DEBUG: PDF file path: {pdf_path}")
        
        # Delete database records
        entities_deleted = db.query(Entity).filter(Entity.document_id == document_id).delete()
        print(f"DEBUG: Deleted {entities_deleted} associated entities")
        
        db.delete(document)
        db.commit()
        print(f"DEBUG: Deleted document record from database")
        
        # Delete files
        files_deleted = []
        files_failed = []
        
        # Delete text file
        success_msg, error_msg = delete_file_safely(text_path, "text file")
        if success_msg:
            files_deleted.append(success_msg)
            print(f"DEBUG: {success_msg}")
        if error_msg:
            files_failed.append(error_msg)
            print(f"ERROR: {error_msg}")
        
        # Delete PDF file
        success_msg, error_msg = delete_file_safely(pdf_path, "PDF file")
        if success_msg:
            files_deleted.append(success_msg)
            print(f"DEBUG: {success_msg}")
        if error_msg:
            files_failed.append(error_msg)
            print(f"ERROR: {error_msg}")
        
        # Prepare response
        response = {
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id,
            "entities_deleted": entities_deleted,
            "files_deleted": files_deleted
        }
        
        if files_failed:
            response["files_failed"] = files_failed
            response["warning"] = "Some files could not be deleted from filesystem"
        
        print(f"DEBUG: Deletion completed. Response: {response}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"ERROR: Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete document: {str(e)}"
        )

@router.get("/documents/{document_id}/pdf")
async def get_document_pdf(document_id: int, db: Session = Depends(get_db)):
    """Serve the original PDF file for viewing in browser."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found in database")
    
    if not document.path_to_text:
        raise HTTPException(status_code=404, detail="No file path information found for this document")
    
    text_path = Path(document.path_to_text)
    pdf_path = determine_pdf_path_from_text_path(text_path)
    
    print(f"DEBUG: Looking for PDF at: {pdf_path}")
    
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"Original PDF file not found: {pdf_path}")
    
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename={document.filename}",
            "Cache-Control": "no-cache"
        }
    )
