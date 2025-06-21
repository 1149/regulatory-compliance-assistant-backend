# File handling utilities
import os
from pathlib import Path
from fastapi import HTTPException
from config import UPLOAD_DIR_NAME

def setup_upload_directory():
    """Create upload directory if it doesn't exist."""
    upload_dir = Path(UPLOAD_DIR_NAME)
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir

def get_file_paths(filename: str, upload_dir: Path):
    """Generate file paths for PDF, text, and raw text files."""
    pdf_path = upload_dir / filename
    text_path = upload_dir / f"{Path(filename).stem}.txt"
    raw_text_path = upload_dir / f"{Path(filename).stem}_raw.txt"
    
    return pdf_path, text_path, raw_text_path

def determine_pdf_path_from_text_path(text_path: Path):
    """Determine PDF path from text path, handling double extensions."""
    if text_path.name.endswith('.txt.txt'):
        base_name = text_path.name[:-8]  # Remove '.txt.txt'
        pdf_filename = f"{base_name}.pdf"
    else:
        pdf_filename = f"{text_path.stem}.pdf"
    
    return text_path.parent / pdf_filename

def delete_file_safely(file_path: Path, file_type: str):
    """Safely delete a file and return status."""
    if file_path and file_path.exists():
        try:
            file_path.unlink()
            return f"Deleted {file_type}: {file_path}", None
        except Exception as e:
            return None, f"{file_type} {file_path}: {str(e)}"
    elif file_path:
        return f"{file_type} not found (already deleted?): {file_path}", None
    return None, None

def find_text_file_paths(document_path_to_text: str):
    """Find possible text file paths for a document."""
    if not document_path_to_text:
        return [], []
    
    processed_path = Path(document_path_to_text)
    
    if processed_path.name.endswith('.txt.txt'):
        base_name = processed_path.name[:-8]  # Remove '.txt.txt'
        
        raw_paths_to_try = [
            processed_path.parent / f"{base_name}_raw.txt",
            processed_path.parent / f"{processed_path.stem}_raw{processed_path.suffix}",
        ]
        
        processed_paths_to_try = [
            processed_path,
            processed_path.parent / f"{base_name}.txt",
        ]
    else:
        raw_paths_to_try = [
            processed_path.parent / f"{processed_path.stem}_raw{processed_path.suffix}",
        ]
        processed_paths_to_try = [
            processed_path,
        ]
    
    return raw_paths_to_try, processed_paths_to_try

def read_text_file(file_path: Path):
    """Read text file content safely."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read file {file_path}: {e}")
