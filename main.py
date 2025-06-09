from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status, Body
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base, Document, DocumentSchema, Entity, EntitySchema 
from pathlib import Path
import os
import httpx
from datetime import datetime
from PyPDF2 import PdfReader
import spacy 
import json  # Ensure json is imported
from typing import Optional  # Ensure Optional is imported if not already

# NEW: Import Google Generative AI client
import google.generativeai as genai
# NEW: Import necessary exceptions for error handling
from google.api_core.exceptions import GoogleAPIError, InvalidArgument, ResourceExhausted

# Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, engine, Base, Document, DocumentSchema

# NEW: Import and load dotenv for environment variables
from dotenv import load_dotenv

# Load environment variables from .env file at the very top level
load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI()

# NEW: Google Gemini Configuration (replaces OpenAI config)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "models/gemini-1.5-flash-latest" # Or another Gemini model if preferred, like gemini-1.5-flash-latest

if not GOOGLE_API_KEY:
    # Raise a clear error if the API key is not found
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please add it to your .env file.")

# Configure the Gemini API client
genai.configure(api_key=GOOGLE_API_KEY)

# Keep Ollama config for the *separate* local endpoint
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
# Check if Ollama env vars are set for the local endpoint to work
if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
    print("Warning: Ollama environment variables (OLLAMA_BASE_URL, OLLAMA_MODEL) are not fully set. Local LLM endpoint might not work.")


# CORS Middleware Configuration (remains the same)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add any other origins where your frontend might be running
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.get("/api/documents/{document_id}/entities", response_model=list[EntitySchema]) # Return list of EntitySchema
async def get_document_entities(document_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to retrieve extracted entities for a specific document.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    entities = db.query(Entity).filter(Entity.document_id == document_id).all()
    return entities    

@app.get("/api/documents/", response_model=list[DocumentSchema])
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
    text_file_location = UPLOAD_DIR / f"{Path(file.filename).stem}.txt"  # e.g., "document.pdf" -> "document.txt"

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
                extracted_text += page.extract_text() or ""

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

    # --- NEW: Extract and Save Entities ---
    extracted_entities_data = []
    try:
        if extracted_text:  # Only run NER if text was successfully extracted
            extracted_entities_data = extract_entities_with_spacy(extracted_text)
    except Exception as e:
        print(f"Warning: Failed to extract entities for {file.filename}: {e}")

    # --- NEW: Generate Embeddings ---
    document_embeddings = None
    if extracted_text:
        try:
            embedding_model = "models/embedding-001"
            response = genai.embed_content(
                model=embedding_model,
                content=extracted_text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            document_embeddings = response['embedding']  # This is a list of floats
            embeddings_json_string = json.dumps(document_embeddings)
        except Exception as e:
            print(f"Warning: Failed to generate embeddings for {file.filename}: {e}")
            embeddings_json_string = None
    else:
        embeddings_json_string = None

    # Save document metadata, entities, and embeddings to the database
    try:
        db_document = Document(
            filename=file.filename,
            upload_date=datetime.utcnow(),
            status="processed_text",
            path_to_text=str(text_file_location),
            embeddings=embeddings_json_string  # NEW: Save the embedding string
        )
        db.add(db_document)

        # Create Entity objects and link them to the document
        for entity_data in extracted_entities_data:
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
            "embeddings_saved": bool(document_embeddings),
            "message": "PDF uploaded, text, entities & embeddings extracted, metadata saved successfully!"
        }
    except Exception as e:
        db.rollback()
        # Clean up files if DB transaction fails
        if pdf_file_location.exists():
            os.remove(pdf_file_location)
        if text_file_location.exists():
            os.remove(text_file_location)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save metadata or embeddings to DB: {e}. Files were cleaned up."
        )

# --- MODIFIED: summarize_text to use Google Gemini ---
@app.post("/api/summarize-text/") # This will now be your primary cloud-based summary endpoint
async def summarize_text(text: str = Body(..., embed=True)):
    """
    Endpoint to summarize text using a cloud-based LLM (Google Gemini).
    """
    # NEW: Print incoming text
    print(f"Received text for summarization (length: {len(text)}): {text[:100]}...")

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            f"Summarize the following document for regulatory compliance: {text}",
            generation_config=genai.types.GenerationConfig(max_output_tokens=500) # Limit summary length
        )
        
        summary = response.text.strip() # Access text directly from response

        if not summary:
            raise HTTPException(status_code=500, detail="Gemini returned an empty summary.")

        return {"summary": summary}
    except InvalidArgument as e: # Catch specific Gemini API errors (e.g., content safety, invalid prompt)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gemini API Invalid Argument: {e}. Check prompt or content for safety guidelines."
        )
    except ResourceExhausted: # Often indicates rate limits or quota issues
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Gemini API quota exceeded or rate limit reached. Please check your Google Cloud Console usage."
        )
    except GoogleAPIError as e: # Catch broader API errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google Gemini API error: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during summarization with Gemini: {e}"
        )

# --- NEW: Separate Endpoint for Local LLM (Ollama) ---
@app.post("/api/summarize-text-local/")
async def summarize_text(text: str = Body(..., embed=True)):
    """

    Endpoint to summarize text using the local Ollama LLM (for demonstration purposes).
    Warning: This can be slow on certain hardware.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Check if Ollama environment variables are set before proceeding
    if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ollama environment variables (OLLAMA_BASE_URL, OLLAMA_MODEL) are not set in .env. Local LLM feature disabled."
        )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Summarize the following document for regulatory compliance: {text}",
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            # Increased timeout for local LLM due to potential slowness
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=180.0) # Timeout increased to 3 minutes
            response.raise_for_status()
            
            response_data = response.json()
            summary = response_data.get("response", "").strip()

            if not summary:
                raise HTTPException(status_code=500, detail="Ollama returned an empty summary.")

            return {"summary": summary}
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not connect to local Ollama server. Ensure it's running at {OLLAMA_BASE_URL}. Error: {e}"
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Local Ollama summarization timed out. Model might be too large for your hardware or prompt too long. Try a shorter text or check Ollama performance."
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Local Ollama API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during summarization with local Ollama: {e}"
        )
@app.get("/api/documents/{document_id}/text", response_model=str) # Return plain text
async def get_document_text(document_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to retrieve the raw text content of a specific document.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found in database")
    
    if not document.path_to_text or not Path(document.path_to_text).exists():
        raise HTTPException(status_code=404, detail=f"Text file not found on server for document ID {document_id}. Path: {document.path_to_text}")
    
    try:
        with open(document.path_to_text, "r", encoding="utf-8") as f:
            text_content = f.read()
        return text_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read text file for document ID {document_id}: {e}")

@app.post("/api/test-ner/")
async def test_ner(text: str = Body(..., embed=True)):
    entities = extract_entities_with_spacy(text)
    return {"text": text, "entities": entities}    

# This loads the model once when the app starts, improving performance.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model 'en_core_web_sm'...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# NEW: Function to extract entities using SpaCy
def extract_entities_with_spacy(text: str):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,  # e.g., ORG, GPE, DATE, PERSON
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })
    return entities

@app.get("/api/search/semantic/", response_model=list[DocumentSchema])
async def semantic_search(query: str, db: Session = Depends(get_db)):
    """
    Performs a semantic search on documents based on the query's meaning.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    query_embedding = None
    try:
        # Use the same embedding model as for documents
        embedding_model = "models/embedding-001"
        response = genai.embed_content(
            model=embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"  # Specify task type for query embeddings
        )
        query_embedding = response['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {e}")

    # Fetch all documents with embeddings
    all_documents = db.query(Document).filter(Document.embeddings.isnot(None)).all()

    results = []
    for doc in all_documents:
        try:
            doc_embedding = json.loads(doc.embeddings)  # Convert stored JSON string back to list of floats

            # Calculate Cosine Similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            query_norm = sum(a * a for a in query_embedding) ** 0.5
            doc_norm = sum(b * b for b in doc_embedding) ** 0.5
            similarity = dot_product / (query_norm * doc_norm) if (query_norm * doc_norm) != 0 else 0

            results.append({"document": doc, "similarity": similarity})
        except Exception as e:
            print(f"Warning: Could not process embedding for document ID {doc.id}: {e}")
            # Skip documents with invalid embeddings

    # Sort results by similarity in descending order
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Return only the document objects, preserving the original schema
    return [doc_res["document"] for doc_res in results]

