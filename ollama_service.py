# Ollama local LLM service
import httpx
from datetime import datetime
from fastapi import HTTPException, status, Body
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

async def generate_local_summary(text: str):
    """Generate summary using local Ollama LLM."""
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Check if Ollama environment variables are set
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
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate", 
                json=payload, 
                timeout=180.0
            )
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

async def analyze_policy_with_ollama(policy_text: str):
    """Analyze policy text using local Ollama LLM as fallback."""
    if not policy_text:
        raise HTTPException(status_code=400, detail="Policy text cannot be empty.")
    
    # Check if Ollama environment variables are set
    if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ollama environment variables not set. Local AI analysis unavailable."
        )

    # Truncate text if too long for local model
    max_chars = 50000  # Smaller limit for local models
    if len(policy_text) > max_chars:
        policy_text = policy_text[:max_chars]
        was_truncated = True
    else:
        was_truncated = False

    analysis_prompt = f"""Analyze this policy document and provide a compliance review.

Policy Document:
{policy_text}

Please provide:
1. Policy Overview: What type of policy is this?
2. Strengths: What works well?
3. Areas for Improvement: What needs work?
4. Recommendations: 3-4 specific actionable steps
5. Priority Level: LOW/MEDIUM/HIGH urgency

Keep the response clear and practical."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": analysis_prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate", 
                json=payload, 
                timeout=300.0  # Longer timeout for analysis
            )
            response.raise_for_status()
            
            response_data = response.json()
            analysis = response_data.get("response", "").strip()

            if not analysis:
                raise HTTPException(status_code=500, detail="Local AI returned empty analysis.")

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
                "analysis": analysis,
                "metadata": {
                    "original_text_length": len(policy_text),
                    "word_count": word_count,
                    "was_truncated": was_truncated,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "overall_compliance_score": overall_score,
                    "ai_service": "Local Ollama (Fallback)",
                    "model": OLLAMA_MODEL
                },
                "recommendations_summary": {
                    "high_priority": "Review the improvement areas identified in the analysis",
                    "next_steps": "Implement the recommended actionable steps"
                },
                "api_limitations": {
                    "text_truncated": was_truncated,
                    "max_characters_supported": max_chars,
                    "note": "Analysis performed using local AI due to quota limits",
                    "suggestion": "Local analysis may be less detailed than cloud-based analysis"
                }
            }
            
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local AI service unavailable. Please try again later or contact support."
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Local AI analysis timed out. Please try with shorter text."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local AI analysis failed: {str(e)}"
        )
