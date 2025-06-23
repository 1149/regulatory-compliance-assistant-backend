# 🔒 Security Features for Public Deployment

This document outlines the security measures implemented to protect against malicious file uploads and other security threats when deploying this application publicly.

## 🛡️ File Upload Security

### 1. **File Type Validation**
- Only allows specific file extensions: `.pdf`, `.txt`, `.doc`, `.docx`
- Cross-checks file extension with actual file content (basic validation)
- No external dependencies required for core validation

### 2. **File Size Limits**
- Maximum file size: 10MB per upload
- Prevents denial-of-service attacks via large files
- Configurable in `config.py`

### 3. **Filename Security**
- Sanitizes filenames to prevent path traversal attacks
- Blocks suspicious filename patterns (`.exe`, `.bat`, `.scr`, etc.)
- Generates secure filenames with timestamps and hashes
- Prevents filename collisions

### 4. **Content Scanning**
- Scans for known malicious file signatures (PE executables, etc.)
- Detects suspicious content patterns (script injections, shell commands)
- Uses heuristic analysis for potential threats

### 5. **Quarantine System**
- Automatically quarantines suspicious files
- Stores metadata about quarantine reasons
- Prevents malicious files from entering the main system

### 6. **Secure File Storage**
- Files stored with unique, timestamped names
- Prevents direct access to uploaded files
- Separates original files from processed text

## 📁 Directory Structure (Secure)
```
uploads/
├── 20250623_a1b2c3d4_document.pdf      # Secure filename format
├── 20250623_a1b2c3d4_document.txt      # Processed text
└── 20250623_a1b2c3d4_document_raw.txt  # Raw extracted text

quarantine/
├── 20250623_e5f6g7h8_suspicious.exe    # Quarantined malicious file
└── 20250623_e5f6g7h8_suspicious.exe.meta  # Quarantine metadata
```

## 🚫 What Gets Blocked

### File Types
- Executable files (`.exe`, `.bat`, `.cmd`, `.scr`)
- Script files (`.js`, `.vbs`, `.ps1`)
- Archive files with executables
- System files (`.dll`, `.sys`)

### Content Patterns
- PE executable headers
- Shell command patterns
- Script injection attempts
- Suspicious binary signatures

### Filename Attacks
- Path traversal attempts (`../`, `..\\`)
- Null byte injections
- Extremely long filenames
- Hidden/system file patterns

## ⚙️ Configuration

In `config.py`:
```python
# Security settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx'}
ENABLE_FILE_QUARANTINE = True
```

## 🔧 Additional Recommendations for Public Deployment

### 1. **Rate Limiting**
```python
# Add to your FastAPI app
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/upload-document/")
@limiter.limit("5/minute")  # 5 uploads per minute per IP
```

### 2. **Authentication (Optional)**
- Consider adding API keys for public use
- Implement user registration/login
- Add usage quotas per user

### 3. **Monitoring & Logging**
- Log all file uploads and security events
- Monitor quarantine directory for threats
- Set up alerts for repeated security violations

### 4. **Infrastructure Security**
- Use HTTPS only
- Set up proper CORS policies
- Consider running in sandboxed containers
- Regular security updates

## 📊 Security Metrics

The system tracks:
- ✅ Files successfully processed
- ⚠️ Files quarantined (with reasons)
- 🚫 Upload attempts blocked
- 📈 Security threat patterns

## 🆘 Incident Response

If malicious content is detected:
1. File is automatically quarantined
2. Security alert is logged
3. Upload is rejected with generic error message
4. IP address can be temporarily blocked (if implemented)

## 🧪 Testing Security

To test the security features:
```bash
# Test file size limit
curl -X POST -F "file=@large_file.pdf" -F "subject=test" http://localhost:8000/api/upload-document/

# Test malicious file detection
curl -X POST -F "file=@malicious.exe" -F "subject=test" http://localhost:8000/api/upload-document/
```

## ⚠️ Known Limitations & API Quotas

### Google Gemini API Quota & Character Limits
The application uses Google Gemini API for AI-powered policy analysis. To ensure reliable public usage and prevent API quota issues, character limits have been implemented:

**Character Limits (Client-Side Prevention):**
- **Maximum:** 100,000 characters per analysis (~25,000 words)
- **Warning:** 80,000+ characters (approaching limit notification)
- **Optimal:** Under 50,000 characters (~12,500 words)

**What Happens with Large Text:**
- Text over 100,000 characters cannot be submitted for analysis
- Users see a clear message: "Content too large - please reduce text size"
- Analysis button is disabled until text is reduced
- Suggests breaking content into smaller sections

**API Quotas (Google Gemini Free Tier):**
- 15 requests per minute
- 1,500 requests per day
- 1 million tokens per day

**Error Handling:**
- Client prevents oversized submissions
- Server validates text length as backup
- Clear user-friendly error messages
- No failed API calls due to size limits

**Solutions for Large Documents:**
1. **Break into Sections:** Analyze chapters or sections separately
2. **Summarize First:** Create a summary under 100k characters
3. **Focus on Key Parts:** Extract and analyze the most important clauses
4. **Use Document Upload:** Upload files and analyze the extracted text in chunks

### Text Length Guidelines
- **Optimal:** Under 50,000 characters (~12,500 words) - Fast analysis
- **Good:** 50,000-80,000 characters (~12,500-20,000 words) - Good performance
- **Warning:** 80,000-100,000 characters (~20,000-25,000 words) - Approaching limit
- **Blocked:** Over 100,000 characters - Cannot analyze, must reduce size

### Rate Limiting for Public Use
For public deployment, consider implementing:
- User-based quotas (X analyses per user per day)
- IP-based rate limiting
- Authentication required for AI features
- Queue system for high-demand periods
- Text length warnings in the UI

Remember: **Security is an ongoing process**. Regularly update dependencies and monitor for new threats.
