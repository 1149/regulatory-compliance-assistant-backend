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
import unicodedata  # Add this at the top with your other imports

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

# --- RAG Chunking Configuration (keep these)
CHUNK_SIZE = 500  # Number of characters per chunk
CHUNK_OVERLAP = 100 # Number of characters to overlap between chunks
# Ensure nlp (SpaCy model) is loaded globally before this function as you have it

def get_text_chunks(text: str):
    """
    Splits text into overlapping chunks using a sliding window approach
    to ensure context is not lost at chunk boundaries.
    """
    if not text:
        return []

    # Use a robust text splitter logic. This is more effective than simple sentence splitting for RAG.
    # We will still use the global CHUNK_SIZE and CHUNK_OVERLAP constants.
    
    # First, let's do a basic sentence-aware join to clean up the text from pdfminer
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    full_text = " ".join(sentences)

    chunks = []
    start_index = 0
    # Slide a window across the text
    while start_index < len(full_text):
        end_index = start_index + CHUNK_SIZE
        chunk = full_text[start_index:end_index]
        chunks.append(chunk)
        
        # The next window starts `overlap` characters back from the end of the current chunk
        next_start = start_index + CHUNK_SIZE - CHUNK_OVERLAP
        
        # If the next start position is the same as the current one, it means the step is too small.
        # To prevent an infinite loop, we must advance the start index by at least one character.
        if next_start <= start_index:
            start_index += 1
        else:
            start_index = next_start

    print(f"DEBUG_CHUNKING: Created {len(chunks)} chunks with size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}.")
    for idx, c in enumerate(chunks[:5]):  # Print first 5 chunks
        print(f"DEBUG_CHUNKING: Chunk {idx}: '{c[:100]}...'")
        
    return chunks

# You'll use this function in your /api/document/{document_id}/qa/ endpoint later.

# --- NEW: Post-process SpaCy entities for compliance domain ---
def post_process_spacy_entities(entities: list):
    processed_entities = []
    for ent in entities:
        text = ent["text"]
        label = ent["label"]

        print(f"DEBUG_POST: Processing entity: '{text[:80]}...' (Initial Label: {label})")
        original_label = label

        # Decision: Do we re-classify ALL entities or just specific types?
        # For this innovative approach, let's re-classify entities that SpaCy might struggle with
        # or those that are particularly important (like sections)

        # We will only re-classify certain types of entities to save on API calls and latency
        # For example, those initially tagged as ORG, or that contain "Section"
        should_reclassify = False
        if label in ["ORG", "MISC", "LAW", "FAC", "GPE"] and (
            "Section" in text or "Policy" in text or "Control" in text or
            "Procedure" in text or "Requirement" in text or
            text.upper() in ["GDPR", "CCPA", "HIPAA", "SOX", "MFA"]
        ):
            should_reclassify = True

        # Add specific checks if needed (e.g., if you only want to reclassify 'Section X.X' that SpaCy gets wrong)
        # For instance: if 'Section' in text and label != 'LAW': should_reclassify = True

        if should_reclassify:
            try:
                gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                reclassify_prompt = f"""
                Given the following text from a policy document, classify it as one of these types: LAW, ORGANIZATION, PERSON, DATE, TIME, LOCATION, MISC, CARDINAL, ORDINAL, or NONE_OF_THE_ABOVE.
                Focus on the primary meaning in a regulatory compliance context.
                If it represents a specific rule, policy, section, or regulation, classify it as LAW.
                Respond with only the chosen label.

                Text: "{text}"
                """
                response = gemini_model.generate_content(
                    reclassify_prompt,
                    generation_config=genai.types.GenerationConfig(max_output_tokens=10) # Keep response short
                )

                new_label_raw = response.text.strip().upper()
                # Basic validation for Gemini's response
                if new_label_raw in ["LAW", "ORGANIZATION", "PERSON", "DATE", "TIME", "LOCATION", "MISC", "CARDINAL", "ORDINAL"]:
                    # Map ORGANIZATION -> ORG, LOCATION -> GPE
                    if new_label_raw == "ORGANIZATION":
                        new_label = "ORG"
                    elif new_label_raw == "LOCATION":
                        new_label = "GPE"
                    else:
                        new_label = new_label_raw # Use directly if it matches SpaCy's labels or our custom ones

                    label = new_label
                    print(f"DEBUG_POST:   LLM changed label from {original_label} to {label} for '{text[:80]}...'")
                else:
                    print(f"DEBUG_POST:   LLM returned unmappable label '{new_label_raw}'. Label remains {original_label} for '{text[:80]}...'")

            except Exception as e:
                print(f"Warning: Failed to reclassify entity '{text[:80]}...' with Gemini: {e}")
                # Keep original label if reclassification fails

        if not should_reclassify or original_label == label:
            # Apply your previous rule-based corrections for specific cases if you want a fallback
            # This is where your previous re.match, text.upper() in [], etc. rules would go
            # For now, let's see how much Gemini handles.
            pass # Gemini should handle what we're targeting now.

        if original_label != label:
            print(f"DEBUG_POST:   Final Label changed from {original_label} to {label} for '{text[:80]}...'")
        else:
            print(f"DEBUG_POST:   Final Label remains {label} for '{text[:80]}...'")

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

        # --- Start of Replacement Block ---
        # 1. Replace sequences of two or more newlines with a double newline (a standard paragraph break).
        extracted_text = re.sub(r'\n\s*\n', '\n\n', extracted_text)

        # 2. Replace single newlines that are likely just line wraps within a sentence with a space.
        # This looks for a lowercase letter, a newline, and another lowercase letter.
        extracted_text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', extracted_text)

        # 3. Replace multiple spaces with a single space to clean up the text.
        extracted_text = re.sub(r' +', ' ', extracted_text)

        # 4. Strip any leading/trailing whitespace from the final text.
        extracted_text = extracted_text.strip()
        # --- End of Replacement Block ---

        # --- MORE ROBUST TEXT CLEANING (for perfect flatness and alignment) ---
        # 1. Normalize Unicode characters (e.g., composite characters to single ones)
        extracted_text = unicodedata.normalize("NFKC", extracted_text)
        # 2. Replace all types of newlines and carriage returns with a single space
        extracted_text = extracted_text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
        # 3. Replace any sequence of whitespace (including multiple spaces) with a single space and strip leading/trailing
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        # --- End of Robust Cleaning ---

        # NEW PRINT: Inspect the final cleaned text before SpaCy/Gemini
        print(f"DEBUG: Final Cleaned Extracted Text (first 500 chars):\n{extracted_text[:500]}...\n---End Final Cleaned Text---")

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
    """
    # --- NEW: Add this block for conversational greetings ---
    normalized_query = query.strip().lower()
    greetings = [
        "hi", "hello", "hey", "greetings", 
        "good morning", "good afternoon", "good evening"
    ]

    if normalized_query in greetings:
        return {"answer": "Hello! How can I assist you with this document today?"}
    # --- End of new block ---

    # The existing RAG logic continues below
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    # --- Retrieval Part ---
    full_document_text = ""
    try:
        with open(document.path_to_text, "r", encoding="utf-8") as f:
            full_document_text = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read document text file: {e}")

    # 1. Chunk the document
    document_chunks = get_text_chunks(full_document_text)
    print(f"DEBUG_RAG: Query: '{query}'")
    print(f"DEBUG_RAG: Document has {len(document_chunks)} chunks.")

    if not document_chunks:
        return {"answer": "I couldn't process the document content to find relevant chunks."}

    # 2. Generate embedding for the user's query
    query_embedding_vector = None
    try:
        query_embedding_model = "models/embedding-001"
        query_embedding_response = genai.embed_content(
            model=query_embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding_vector = query_embedding_response['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {e}")

    # 3. Generate embeddings for each chunk and find most relevant
    chunk_similarities = []
    for idx, chunk in enumerate(document_chunks):
        try:
            chunk_embedding_response = genai.embed_content(
                model=query_embedding_model,
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT"
            )
            chunk_embedding_vector = chunk_embedding_response['embedding']

            dot_product = sum(a * b for a, b in zip(query_embedding_vector, chunk_embedding_vector))
            query_norm = sum(a*a for a in query_embedding_vector)**0.5
            chunk_norm = sum(b*b for b in chunk_embedding_vector)**0.5

            similarity = dot_product / (query_norm * chunk_norm) if (query_norm * chunk_norm) != 0 else 0

            chunk_similarities.append({"chunk": chunk, "similarity": similarity, "index": idx})

            # NEW DEBUG: Print each chunk's full text and similarity score
            print(f"DEBUG_RAG_CHUNK_DETAIL: Chunk {idx} (Score: {similarity:.4f}): '{chunk[:150]}...'")

        except Exception as e:
            print(f"Warning: Failed to embed or compare chunk {idx}: {e}")
            # Continue to next chunk if one fails

    # NEW DEBUG: Print all chunk similarities before sorting
    top5 = [f"Chunk {s['index']} ({s['similarity']:.4f})" for s in sorted(chunk_similarities, key=lambda x: x['similarity'], reverse=True)[:5]]
    print(f"DEBUG_RAG: All chunk similarities (top 5):\n{top5}")
    chunk_similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_n_chunks = 7  # You can adjust this number

    relevant_chunks = [item["chunk"] for item in chunk_similarities[:top_n_chunks]]

    # NEW DEBUG: Print selected relevant chunks
    print(f"DEBUG_RAG: Top {top_n_chunks} relevant chunks selected:")
    for idx, chunk_text in enumerate(relevant_chunks):
        print(f"DEBUG_RAG:   Chunk {idx+1}: '{chunk_text[:100]}...' (Score: {chunk_similarities[idx]['similarity']:.4f})")
    print("--- End Relevant Chunks ---")

    if not relevant_chunks:
        return {"answer": "I couldn't find any highly relevant sections in the document for your question."}

    # --- Generation Part (Gemini RAG) ---
    try:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        # Construct the RAG prompt for Gemini
        context = "\n\n".join(relevant_chunks)

        rag_prompt = f"""
        You are a helpful assistant that answers questions based ONLY on the provided document context.
        If the answer cannot be found in the context, state that clearly and do not make up information.

        Document Context:
        ---
        {context}
        ---

        Question: {query}
        """
        # NEW DEBUG: Print the final RAG prompt
        print(f"DEBUG_RAG: Final RAG Prompt sent to Gemini:\n{rag_prompt[:500]}...\n--- End RAG Prompt ---")

        # Call Gemini API
        response = gemini_model.generate_content(
            rag_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=500)
        )

        answer = response.text.strip()

        if not answer:
            return {"answer": "The AI model returned an empty response."}

        return {"answer": answer}

    except InvalidArgument as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gemini API Invalid Argument for RAG: {e}. Check prompt or content safety guidelines."
        )
    except ResourceExhausted:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Gemini API quota exceeded for RAG. Please check your Google Cloud Console usage."
        )
    except GoogleAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google Gemini API error during RAG generation: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during RAG answer generation: {e}"
        )

@app.post("/api/test-chunking/")
async def test_chunking(text: str = Body(..., embed=True)):
    chunks = get_text_chunks(text)
    return {"num_chunks": len(chunks), "first_chunks": chunks[:3]}  # Return first 3 chunks

@app.post("/api/analyze-policy/")
async def analyze_policy(policy_text: str = Body(..., embed=True)):
    """
    Endpoint for analyzing user-provided policy text against regulatory clauses.
    For now, returns a placeholder response.
    """
    if not policy_text.strip():
        raise HTTPException(status_code=400, detail="Policy text cannot be empty for analysis.")

    # In future steps, this is where you'd implement:
    # 1. Text extraction (already done for docs, here it's direct text input)
    # 2. Embedding generation for policy_text
    # 3. Semantic search against COMPLIANCE_CLAUSE entities from your DB
    # 4. LLM call (Gemini) to draft suggestions based on matches/gaps

    placeholder_response = {
        "status": "Analysis pending",
        "message": "Policy received for analysis. AI suggestions will be generated soon!",
        "received_text_length": len(policy_text)
    }
    return placeholder_response