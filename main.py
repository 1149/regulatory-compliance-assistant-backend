from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from pathlib import Path
import os
from datetime import datetime

# Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, engine, Base, Document


Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS Middleware Configuration
origins = [
    "http://localhost:3000",  # Your React app's default address
    "http://127.0.0.0.1:3000",  # Another common address for localhost
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

UPLOAD_DIR = Path("uploads")
if not UPLOAD_DIR.exists():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/upload-document/")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db) # NEW: Get database session
):
    try:
        # Create a unique file path for the uploaded file
        file_location = UPLOAD_DIR / file.filename
        
        # Save the uploaded file to disk
        with open(file_location, "wb+") as file_object:
            content = await file.read()
            file_object.write(content)

        # NEW: Save document metadata to the database
        db_document = Document(
            filename=file.filename,
            upload_date=datetime.utcnow(), # Set current UTC time
            status="pending_processing",  # Initial status
            path_to_text=str(file_location) # Store the file's path
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document) # Refresh to get the generated ID and updated fields

        return {
            "filename": db_document.filename,
            "location": db_document.path_to_text,
            "id": db_document.id, # Return the ID from the database
            "message": "File uploaded and metadata saved successfully!"
        }
    except Exception as e:
        db.rollback() # Rollback changes if an error occurs
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload or save metadata: {e}"
        )