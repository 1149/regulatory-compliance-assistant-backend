# Gemini AI service functions
import json
from datetime import datetime
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, InvalidArgument, ResourceExhausted
from fastapi import HTTPException, status
from config import GEMINI_MODEL_NAME, GOOGLE_API_KEY

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini API Limits for gemini-1.5-flash
MAX_INPUT_TOKENS = 1048576  # ~1M tokens (approximately 750K-800K characters)
MAX_CHARS_SAFE_LIMIT = 600000  # 600K characters for safety margin
MAX_WORDS_SAFE_LIMIT = 100000  # ~100K words for safety

def truncate_text_safely(text: str, max_chars: int = MAX_CHARS_SAFE_LIMIT) -> tuple[str, bool]:
    """
    Safely truncate text to fit within API limits.
    Returns (truncated_text, was_truncated)
    """
    if len(text) <= max_chars:
        return text, False
    
    # Try to truncate at a sentence boundary near the limit
    truncated = text[:max_chars]
    last_sentence = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?')
    )
    
    if last_sentence > max_chars * 0.8:  # If we found a sentence ending in the last 20%
        truncated = truncated[:last_sentence + 1]
    
    return truncated, True

def get_text_stats(text: str) -> dict:
    """Get basic statistics about the text."""
    return {
        "character_count": len(text),
        "word_count": len(text.split()),
        "estimated_tokens": len(text) // 4,  # Rough estimate: 4 chars per token
        "within_limits": len(text) <= MAX_CHARS_SAFE_LIMIT
    }

def identify_compliance_clauses(text: str, filename: str):
    """Identify compliance clauses using Gemini AI."""
    identified_clauses_data = []    
    if not text:
        return identified_clauses_data
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        clause_prompt = f"""You are an expert compliance analyst specialized in identifying regulatory clauses and policy statements.

TASK: Extract all compliance-related clauses, rules, and policy statements from the document below.

WHAT TO IDENTIFY:
• Regulatory requirements and obligations
• Compliance policies and procedures  
• Security controls and measures
• Data protection and privacy rules
• Risk management statements
• Audit and monitoring requirements
• Legal and contractual obligations
• Standards and framework references (ISO, NIST, etc.)
• Violation consequences and penalties

FORMAT REQUIREMENTS:
- Each clause must start with "- " (dash and space)
- Include complete, meaningful sentences
- Preserve exact wording when possible
- One clause per line
- If no clear compliance clauses exist, respond with "None"

DOCUMENT TO ANALYZE:
{text}

IDENTIFIED COMPLIANCE CLAUSES:"""
        response = model.generate_content(
            clause_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=1000)
        )
        
        clauses_raw_text = response.text.strip()
        print(f"Gemini raw clause response for {filename}:\n{clauses_raw_text}\n---")

        if clauses_raw_text and clauses_raw_text.lower() != "none":
            clauses_list = [c.strip() for c in clauses_raw_text.split('\n') if c.strip().startswith('- ')]
            print(f"DEBUG: Parsed clauses_list: {clauses_list}")

            for clause_text in clauses_list:
                clean_clause_text = clause_text[2:].strip()
                print(f"DEBUG: Saving clause '{clean_clause_text[:50]}...' with start/end -1")

                identified_clauses_data.append({
                    "text": clean_clause_text,
                    "label": "COMPLIANCE_CLAUSE",
                    "start_char": -1,
                    "end_char": -1
                })

            print(f"DEBUG: identified_clauses_data after loop: {identified_clauses_data}")

    except InvalidArgument as e:
        print(f"Warning: Gemini API Invalid Argument for compliance clause identification in {filename}: {e}")
    except ResourceExhausted:
        print(f"🚨 Google Gemini API quota exceeded! This app uses a FREE API plan with only 50 requests per day. "
              f"The quota resets daily. Please try again tomorrow, or you can get your own free API key at: "
              f"https://makersuite.google.com/app/apikey")
    except GoogleAPIError as e:
        print(f"Warning: Google Gemini API error during compliance clause identification in {filename}: {e}")
    except Exception as e:
        print(f"Warning: Failed to identify compliance clauses for {filename} with Gemini: {e}")
    
    return identified_clauses_data

def generate_document_embeddings(text: str, filename: str):
    """Generate embeddings for document text."""
    if not text:
        return None
    
    try:
        embedding_model = "models/embedding-001"
        response = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        document_embeddings = response['embedding']
        return json.dumps(document_embeddings)
    except InvalidArgument as e:
        print(f"Warning: Gemini API Invalid Argument for document embeddings in {filename}: {e}")
        return None
    except ResourceExhausted:
        print(f"🚨 Google Gemini API quota exceeded! This app uses a FREE API plan with only 50 requests per day. "
              f"The quota resets daily. Please try again tomorrow, or you can get your own free API key at: "
              f"https://makersuite.google.com/app/apikey")
        return None
    except GoogleAPIError as e:
        print(f"Warning: Google Gemini API error during document embeddings for {filename}: {e}")
        return None
    except Exception as e:
        print(f"Warning: Failed to generate embeddings for {filename}: {e}")
        return None

def generate_query_embedding(query: str):
    """Generate embedding for user query."""
    try:
        query_embedding_model = "models/embedding-001"
        query_embedding_response = genai.embed_content(
            model=query_embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        return query_embedding_response['embedding']
    except InvalidArgument as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gemini API Invalid Argument for query embedding: {e}"
        )
    except ResourceExhausted:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="🚨 Google Gemini API quota exceeded! This app uses a FREE API plan with only 50 requests per day. "
                   "The quota resets daily. Please try again tomorrow, or you can get your own free API key at: "
                   "https://makersuite.google.com/app/apikey"
        )
    except GoogleAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google Gemini API error during query embedding: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {e}")

def generate_chunk_embedding(chunk: str):
    """Generate embedding for text chunk."""
    try:
        chunk_embedding_response = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return chunk_embedding_response['embedding']
    except InvalidArgument as e:
        print(f"Warning: Gemini API Invalid Argument for chunk embedding: {e}")
        return None
    except ResourceExhausted:
        print(f"🚨 Google Gemini API quota exceeded! This app uses a FREE API plan with only 50 requests per day. "
              f"The quota resets daily. Please try again tomorrow, or you can get your own free API key at: "
              f"https://makersuite.google.com/app/apikey")
        return None
    except GoogleAPIError as e:
        print(f"Warning: Google Gemini API error during chunk embedding: {e}")
        return None
    except Exception:
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(a*a for a in embedding1)**0.5
    norm2 = sum(b*b for b in embedding2)**0.5
    
    return dot_product / (norm1 * norm2) if (norm1 * norm2) != 0 else 0

def generate_summary(text: str):
    """Generate summary using Gemini."""
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            f"Summarize the following document for regulatory compliance: {text}",
            generation_config=genai.types.GenerationConfig(max_output_tokens=500)
        )
        
        summary = response.text.strip()
        if not summary:
            raise HTTPException(status_code=500, detail="Gemini returned an empty summary.")
        
        return {"summary": summary}
    
    except InvalidArgument as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gemini API Invalid Argument: {e}. Check prompt or content for safety guidelines."
        )
    except ResourceExhausted:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="🚨 Google Gemini API quota exceeded! This app uses a FREE API plan with only 50 requests per day. "
                   "The quota resets daily. Please try again tomorrow, or you can get your own free API key at: "
                   "https://makersuite.google.com/app/apikey"
        )
    except GoogleAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google Gemini API error: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during summarization with Gemini: {e}"
        )

def generate_rag_answer(context: str, query: str):
    """Generate RAG-based answer using Gemini."""
    try:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        rag_prompt = f"""
        You are a helpful assistant that answers questions based ONLY on the provided document context.
        If the answer cannot be found in the context, state that clearly and do not make up information.

        Document Context:
        ---
        {context}
        ---

        Question: {query}
        """
        
        print(f"DEBUG_RAG: Final RAG Prompt sent to Gemini:\n{rag_prompt[:500]}...\n--- End RAG Prompt ---")

        response = gemini_model.generate_content(
            rag_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=500)
        )

        answer = response.text.strip()
        if not answer:
            return "The AI model returned an empty response."
        
        return answer

    except InvalidArgument as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gemini API Invalid Argument for RAG: {e}. Check prompt or content safety guidelines."
        )
    except ResourceExhausted:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="🚨 Google Gemini API quota exceeded! This app uses a FREE API plan with only 50 requests per day. "
                   "The quota resets daily. Please try again tomorrow, or you can get your own free API key at: "
                   "https://makersuite.google.com/app/apikey"
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

async def analyze_policy_text(policy_text: str):
    """Analyze policy text using Gemini AI."""
    try:
        # Truncate policy text to stay within safe character limits
        truncated_policy_text, was_truncated = truncate_text_safely(policy_text)
        if was_truncated:
            print(f"Warning: Policy text for analysis was truncated to {MAX_CHARS_SAFE_LIMIT} characters.")

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        analysis_prompt = f"""
        You are a friendly compliance expert helping businesses improve their policies. Analyze this policy document and provide a clear, easy-to-understand review.

        Policy Document:
        ---
        {truncated_policy_text}
        ---

        Please provide your analysis in this simple, conversational format:

        ## What This Policy Is About
        Briefly describe what type of policy this is and its main purpose (2-3 sentences).

        ## Overall Assessment
        Give an honest, straightforward assessment of the policy's current state. Is it comprehensive, basic, or needs significant work? (2-3 sentences)

        ## What's Working Well
        List 3-4 specific strengths of this policy. What does it do right?

        ## What Needs Improvement
        List the top 4-5 most important areas that need attention. Be specific and practical:
        - Area 1: Brief description and why it matters
        - Area 2: Brief description and why it matters  
        - Area 3: Brief description and why it matters
        (etc.)

        ## Quick Win Recommendations
        Suggest 3-4 specific, actionable steps they can take immediately to improve the policy:
        1. [Specific action]
        2. [Specific action]
        3. [Specific action]

        ## Priority Level
        Rate the urgency of updating this policy as: LOW, MEDIUM, or HIGH and explain why in 1-2 sentences.

        Keep the language simple, practical, and encouraging. Focus on actionable advice rather than technical compliance jargon.
        """

        response = model.generate_content(
            analysis_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=1500)
        )
        
        analysis_result = response.text.strip()
        if not analysis_result:
            raise HTTPException(status_code=500, detail="AI analysis returned empty response.")

        # Calculate basic score based on policy length
        word_count = len(policy_text.split())
        if word_count < 100:
            overall_score = 3.0
        elif word_count < 300:
            overall_score = 5.0
        elif word_count < 800:
            overall_score = 7.0
        else:
            overall_score = 8.5

        return {
            "status": "completed",
            "analysis": analysis_result,
            "metadata": {
                "original_text_length": len(policy_text),
                "processed_text_length": len(truncated_policy_text),
                "word_count": word_count,
                "was_truncated": was_truncated,
                "character_limit": MAX_CHARS_SAFE_LIMIT,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "overall_compliance_score": overall_score
            },
            "recommendations_summary": {
                "high_priority": "Review the specific improvement areas identified in the analysis",
                "next_steps": "Start with the 'Quick Win Recommendations' to make immediate improvements"
            },
            "api_limitations": {
                "text_truncated": was_truncated,
                "max_characters_supported": MAX_CHARS_SAFE_LIMIT,
                "truncation_message": f"Your policy text was {len(policy_text):,} characters. Only the first {len(truncated_policy_text):,} characters were analyzed." if was_truncated else None,
                "suggestion": "For complete analysis of very large documents, consider breaking them into smaller sections." if was_truncated else None
            }
        }

    except InvalidArgument as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gemini API Invalid Argument for policy analysis: {e}"
        )
    except ResourceExhausted:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="🚨 Google Gemini API quota exceeded! This app uses a FREE API plan with only 50 requests per day. "
                   "The quota resets daily. Please try again tomorrow, or you can get your own free API key at: "
                   "https://makersuite.google.com/app/apikey"
        )
    except GoogleAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google Gemini API error during policy analysis: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during policy analysis: {e}"
        )
