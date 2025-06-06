from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from pathlib import Path
import os
from datetime import datetime
from PyPDF2 import PdfReader # Ensure this import is present at the top

# Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, engine, Base, Document, DocumentSchema


Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS Middleware Configuration
origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",  
    # Add any other origins where your frontend might be running
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/test-db")
async def test_db_connection(db: Session = Depends(get_db)):
    try:
        db.query(Document).first()
        return {"message": "Database connection successful!"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection failed: {e}"
        )
@app.get("/api/documents/", response_model=list[DocumentSchema]) # Add response_model for better API docs
async def get_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).all()
    return documents

UPLOAD_DIR = Path("uploads")
if not UPLOAD_DIR.exists():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/upload-document/")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Added validation for PDF files
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed."
        )

    # Define paths for the uploaded PDF and the extracted text file
    pdf_file_location = UPLOAD_DIR / file.filename
    text_file_location = UPLOAD_DIR / f"{Path(file.filename).stem}.txt" # e.g., "document.pdf" -> "document.txt"

    # Save the uploaded PDF file to disk
    try:
        with open(pdf_file_location, "wb+") as file_object:
            content = await file.read()
            file_object.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save PDF file: {e}"
        )

    # Extract text from PDF
    extracted_text = ""
    try:
        with open(pdf_file_location, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text() or "" # extract_text might return None
        
        # Save the extracted text to a .txt file
        with open(text_file_location, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)

    except Exception as e:
        # If text extraction fails, attempt to delete the PDF and rollback DB transaction if it started
        if pdf_file_location.exists():
            os.remove(pdf_file_location)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract text from PDF: {e}. Please ensure the PDF is not password-protected or corrupted."
        )

    # Save document metadata to the database
    try:
        db_document = Document(
            filename=file.filename,
            upload_date=datetime.utcnow(),
            status="processed_text", # Change status to indicate text extraction
            path_to_text=str(text_file_location) # Store the path to the .txt file
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document) # Refresh to get the generated ID and updated fields

        return {
            "filename": db_document.filename,
            "text_location": db_document.path_to_text, # Return text file location
            "id": db_document.id,
            "message": "PDF uploaded, text extracted, and metadata saved successfully!"
        }
    except Exception as e:
        db.rollback() # Rollback changes if a database error occurs
        # Clean up files if DB transaction fails
        if pdf_file_location.exists():
            os.remove(pdf_file_location)
        if text_file_location.exists():
            os.remove(text_file_location)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save metadata to DB: {e}. Files were cleaned up."
        )