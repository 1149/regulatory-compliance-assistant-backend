# AI and analysis routes
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import SessionLocal, Document
from text_utils import get_text_chunks
from ai_services import (
    generate_summary, 
    generate_query_embedding, 
    generate_chunk_embedding, 
    calculate_similarity,
    generate_rag_answer,
    analyze_policy_text
)
from ollama_service import generate_local_summary
from nlp_utils import extract_entities_with_spacy

router = APIRouter(prefix="/api", tags=["ai"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/summarize-text/")
async def summarize_text(text: str = Body(..., embed=True)):
    """Endpoint to summarize text using Google Gemini."""
    print(f"Received text for summarization (length: {len(text)}): {text[:100]}...")
    return generate_summary(text)

@router.post("/summarize-text-local/")
async def summarize_text_local(text: str = Body(..., embed=True)):
    """Endpoint to summarize text using local Ollama LLM."""
    return await generate_local_summary(text)

@router.post("/test-ner/")
async def test_ner(text: str = Body(..., embed=True)):
    """Test NER functionality on provided text."""
    entities = extract_entities_with_spacy(text)
    return {"text": text, "entities": entities}

@router.post("/test-chunking/")
async def test_chunking(text: str = Body(..., embed=True)):
    """Test text chunking functionality."""
    chunks = get_text_chunks(text)
    return {"num_chunks": len(chunks), "first_chunks": chunks[:3]}

@router.post("/document/{document_id}/qa/")
async def document_qa(document_id: int, query: str = Body(..., embed=True), db: Session = Depends(get_db)):
    """AI Chatbot endpoint for Question-Answering over a specific document (RAG)."""
    
    # Handle greetings
    normalized_query = query.strip().lower()
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    
    if normalized_query in greetings:
        return {"answer": "Hello! How can I assist you with this document today?"}

    # Get document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Read document text
    try:
        with open(document.path_to_text, "r", encoding="utf-8") as f:
            full_document_text = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read document text file: {e}")

    # Chunk the document
    document_chunks = get_text_chunks(full_document_text)
    print(f"DEBUG_RAG: Query: '{query}'")
    print(f"DEBUG_RAG: Document has {len(document_chunks)} chunks.")

    if not document_chunks:
        return {"answer": "I couldn't process the document content to find relevant chunks."}

    # Generate query embedding
    query_embedding_vector = generate_query_embedding(query)

    # Find most relevant chunks
    chunk_similarities = []
    for idx, chunk in enumerate(document_chunks):
        chunk_embedding_vector = generate_chunk_embedding(chunk)
        if chunk_embedding_vector:
            similarity = calculate_similarity(query_embedding_vector, chunk_embedding_vector)
            chunk_similarities.append({"chunk": chunk, "similarity": similarity, "index": idx})
            print(f"DEBUG_RAG_CHUNK_DETAIL: Chunk {idx} (Score: {similarity:.4f}): '{chunk[:150]}...'")
        else:
            print(f"Warning: Failed to embed chunk {idx}")

    # Sort by similarity and get top chunks
    chunk_similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_n_chunks = 7
    relevant_chunks = [item["chunk"] for item in chunk_similarities[:top_n_chunks]]

    print(f"DEBUG_RAG: Top {top_n_chunks} relevant chunks selected:")
    for idx, chunk_text in enumerate(relevant_chunks):
        score = chunk_similarities[idx]['similarity'] if idx < len(chunk_similarities) else 0
        print(f"DEBUG_RAG:   Chunk {idx+1}: '{chunk_text[:100]}...' (Score: {score:.4f})")

    if not relevant_chunks:
        return {"answer": "I couldn't find any highly relevant sections in the document for your question."}

    # Generate answer using RAG
    context = "\n\n".join(relevant_chunks)
    answer = generate_rag_answer(context, query)
    
    return {"answer": answer}

@router.post("/analyze-policy/")
async def analyze_policy(policy_text: str = Body(..., embed=True)):
    """Comprehensive endpoint for analyzing user-provided policy text."""
    if not policy_text.strip():
        raise HTTPException(status_code=400, detail="Policy text cannot be empty for analysis.")
    
    return analyze_policy_text(policy_text)
