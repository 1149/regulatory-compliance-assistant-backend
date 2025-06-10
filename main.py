from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status, Body
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base, Document, DocumentSchema, Entity, EntitySchema
from pathlib import Path
import os
import httpx
from datetime import datetime
from pdfminer.high_level import extract_text as pdfminer_extract_text
import spacy
import json
from rapidfuzz import process, fuzz
from typing import Optional
import re # Add this at the top with your other imports

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

# --- SpaCy Initialization (move here) ---
try:
    nlp = spacy.load("en_core_web_md") # UPGRADED: Use medium model for better NER
except OSError:
    print("Downloading SpaCy model 'en_core_web_md'...")
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# --- Function to extract entities using SpaCy (move here) ---
def extract_entities_with_spacy(text: str):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_, # e.g., ORG, GPE, DATE, PERSON
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })
    return entities

# --- NEW: Post-process SpaCy entities for compliance domain ---
def post_process_spacy_entities(entities: list):
    processed_entities = []
    for ent in entities:
        text = ent["text"]
        label = ent["label"]

        print(f"DEBUG_POST: Processing entity: '{text[:80]}...' (Initial Label: {label})")
        original_label = label

        # --- ALWAYS PRINT repr(text) and string comparison results ---
        print(f"DEBUG_ULTRA: repr(text): {repr(text)}")
        print(f"DEBUG_ULTRA: 'Access Control' in text: {('Access Control' in text)}")
        print(f"DEBUG_ULTRA: 'Section' in text: {('Section' in text)}")
        print(f"DEBUG_ULTRA: Combined condition: {('Access Control' in text and 'Section' in text)}")

        # --- NEW HIGH-PRIORITY, DIRECT RULE ---
        if "Access Control" in text and "Section" in text:
            label = "LAW"
            print(f"DEBUG_ULTRA: Label *should have* changed to LAW for '{text[:80]}...'")
        # --- END NEW RULE ---
        
        # RULE 1 (now elif): Classify general 'Section' or 'Policy' headers as LAW
        elif re.match(r'^(Section|Policy)\s+(\d+(\.\d+)*)?\s*[-:]?\s*.*', text, re.IGNORECASE):
            label = "LAW"
        # RULE 2 (now elif): Correct common miscategorizations for well-known compliance terms
        elif text.upper() in ["GDPR", "CCPA", "HIPAA", "SOX"]:
            label = "LAW"
        # RULE 3 (now elif): Catch other specific miscategorizations as LAW if they are ORG and contain compliance keywords
        elif label == "ORG" and any(keyword in text for keyword in ["Data Collection", "Access Control", "Incident Response Procedures", "Compliance and Audit"]):
            label = "LAW"
        # RULE 4 (now elif): General corrections for entities that are clearly not ORG/PERSON/LOC etc.
        elif label == "ORG" and "Public Data" in text:
            label = "MISC"
        elif label == "ORG" and "MFA" in text:
            label = "MISC"
        elif label == "WORK_OF_ART" and "Confidential" in text:
            label = "MISC"

        # Add more rules as you observe consistent miscategorizations specific to your documents

        # NEW DEBUG PRINT: After processing rules
        if original_label != label:
            print(f"DEBUG_POST:   Label changed from {original_label} to {label} for '{text[:80]}...'")
        else:
            print(f"DEBUG_POST:   Label remains {label} for '{text[:80]}...'")

        processed_entities.append({
            "text": text,
            "label": label,
            "start_char": ent["start_char"],
            "end_char": ent["end_char"]
        })
    return processed_entities

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
    text_file_location = UPLOAD_DIR / f"{Path(file.filename).stem}.txt"

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

    # --- MODIFIED: Extract text from PDF using pdfminer.six ---
    extracted_text = ""
    try:
        # Use pdfminer_extract_text directly
        extracted_text = pdfminer_extract_text(str(pdf_file_location))

        # --- MORE ROBUST TEXT CLEANING (using regex) ---
        # Replace all whitespace sequences (including newlines, tabs) with a single space
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()

        # NEW PRINT: Inspect the cleaned text before SpaCy/Gemini
        print(f"DEBUG: Cleaned Extracted Text (first 500 chars):\n{extracted_text[:500]}...\n---End Cleaned Text---")

        # Save the extracted text to a .txt file
        with open(text_file_location, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)

    except Exception as e:
        if pdf_file_location.exists():
            os.remove(pdf_file_location)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract text from PDF using pdfminer.six: {e}. Please ensure the PDF is not password-protected or corrupted."
        )

    # --- Identify Compliance Clauses using Gemini ---
    identified_clauses_data = []
    if extracted_text:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            clause_prompt = f"""
            From the following document, identify and list any distinct paragraphs or sentences that appear to be specific regulatory clauses, policies, or compliance rules.
            List each identified clause on a new line, prefixed with "- ".
            If no such clauses are clearly identifiable, respond with "None".

            Document:
            {extracted_text}
            """
            response = model.generate_content(
                clause_prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=1000)
            )
            
            clauses_raw_text = response.text.strip()
            print(f"Gemini raw clause response for {file.filename}:\n{clauses_raw_text}\n---")

            if clauses_raw_text and clauses_raw_text.lower() != "none":
                clauses_list = [c.strip() for c in clauses_raw_text.split('\n') if c.strip().startswith('- ')]
                print(f"DEBUG: Parsed clauses_list: {clauses_list}") # Debug print

                for clause_text in clauses_list:
                    clean_clause_text = clause_text[2:].strip()

                    # --- REPLACEMENT: Always save clause, bypass fuzzy matching ---
                    start = -1
                    end = -1
                    print(f"DEBUG: Saving clause '{clean_clause_text[:50]}...' with start/end -1 (bypassing fuzzy match error)")
                    # --- END OF REPLACEMENT ---

                    identified_clauses_data.append({
                        "text": clean_clause_text,
                        "label": "COMPLIANCE_CLAUSE",
                        "start_char": start,
                        "end_char": end
                    })

                print(f"DEBUG: identified_clauses_data after loop: {identified_clauses_data}") # Keep this debug print

        except Exception as e:
            print(f"Warning: Failed to identify compliance clauses for {file.filename} with Gemini: {e}")
            identified_clauses_data = []

    # --- Extract and Save Entities (SpaCy) ---
    extracted_entities_data_spacy = []
    try:
        if extracted_text:
            extracted_entities_data_spacy = extract_entities_with_spacy(extracted_text)
            # Post-process SpaCy entities (this line should already be here)
            extracted_entities_data_spacy = post_process_spacy_entities(extracted_entities_data_spacy)

        # MOVED PRINT: Now inspect SpaCy's entities AFTER post-processing
        print(f"DEBUG: SpaCy Extracted Entities (Post-Processed):\n{extracted_entities_data_spacy}\n---End SpaCy Entities---")

    except Exception as e:
        print(f"Warning: Failed to extract SpaCy entities for {file.filename}: {e}")
        extracted_entities_data_spacy = []

    # --- Generate Embeddings ---
    document_embeddings = None
    if extracted_text:
        try:
            embedding_model = "models/embedding-001"
            response = genai.embed_content(
                model=embedding_model,
                content=extracted_text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            document_embeddings = response['embedding']
            embeddings_json_string = json.dumps(document_embeddings)
        except Exception as e:
            print(f"Warning: Failed to generate embeddings for {file.filename}: {e}")
            embeddings_json_string = None
    else:
        embeddings_json_string = None

    # Save document metadata, entities, and clauses to the database
    try:
        db_document = Document(
            filename=file.filename,
            upload_date=datetime.utcnow(),
            status="processed_text",
            path_to_text=str(text_file_location),
            embeddings=embeddings_json_string
        )
        db.add(db_document)

        # Combine SpaCy entities and identified clauses
        all_entities_to_save = extracted_entities_data_spacy + identified_clauses_data

        for entity_data in all_entities_to_save:
            # Debug print to see what is being saved
            print("Saving entity:", entity_data)
            # Skip entities with invalid char positions
            # if entity_data["start_char"] == -1 or entity_data["end_char"] == -1:
            #     print(f"Skipping entity with invalid char positions: {entity_data}")
            #     continue
            # Optionally, check text length here if your DB column is limited
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
            "entities_identified": len(all_entities_to_save),
            "message": "PDF uploaded, text, entities & clauses extracted, metadata saved successfully!"
        }
    except Exception as e:
        db.rollback()
        if pdf_file_location.exists():
            os.remove(pdf_file_location)
        if text_file_location.exists():
            os.remove(text_file_location)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save metadata or entities/clauses to DB: {e}. Files were cleaned up."
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

@app.post("/api/document/{document_id}/qa/")
async def document_qa(document_id: int, query: str = Body(..., embed=True), db: Session = Depends(get_db)):
    """
    AI Chatbot endpoint for Question-Answering over a specific document (RAG).
    For now, returns a placeholder.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    # For now, just return a placeholder response
    placeholder_answer = f"Hello! You asked about document ID {document_id}: '{query}'. I will generate an answer for you soon!"
    return {"answer": placeholder_answer}