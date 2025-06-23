# Security utilities for file upload validation
import os
import re
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException, status

from config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS, QUARANTINE_DIR

# Known malicious file signatures (magic bytes)
MALICIOUS_SIGNATURES = [
    b'\x4d\x5a',  # PE executable (MZ header)
    b'\x50\x4b\x03\x04',  # ZIP with executable content (common for malware)
    b'\x7f\x45\x4c\x46',  # ELF executable
    b'\xcf\xfa\xed\xfe',  # Mach-O executable
    b'\xfe\xed\xfa\xce',  # Mach-O executable (reverse)
]

# Suspicious filename patterns
SUSPICIOUS_PATTERNS = [
    r'\.exe$', r'\.bat$', r'\.cmd$', r'\.scr$', r'\.com$',
    r'\.pif$', r'\.vbs$', r'\.js$', r'\.jar$', r'\.dll$',
    r'\.sys$', r'\.msi$', r'\.app$', r'\.deb$', r'\.rpm$',
    r'\.dmg$', r'\.pkg$', r'\.iso$', r'\.img$'
]

def setup_quarantine_directory():
    """Create quarantine directory if it doesn't exist."""
    quarantine_dir = Path(QUARANTINE_DIR)
    if not quarantine_dir.exists():
        quarantine_dir.mkdir(parents=True, exist_ok=True)
    return quarantine_dir

def validate_filename(filename: str) -> Tuple[bool, str]:
    """
    Validate filename for security issues.
    Returns (is_safe, reason)
    """
    if not filename:
        return False, "Empty filename"
    
    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Path traversal attempt detected"
    
    # Check for suspicious patterns
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return False, f"Suspicious file extension detected: {filename}"
    
    # Check filename length
    if len(filename) > 255:
        return False, "Filename too long"
    
    # Check for null bytes
    if '\x00' in filename:
        return False, "Null byte in filename"
    
    return True, "Valid filename"

def validate_basic_file_type(file_content: bytes, filename: str) -> Tuple[bool, str]:
    """
    Basic file type validation without external libraries.
    Returns (is_safe, reason)
    """
    file_extension = Path(filename).suffix.lower()
    
    # PDF validation
    if file_extension == '.pdf':
        if not file_content.startswith(b'%PDF-'):
            return False, "Invalid PDF file structure"
    
    # Text file validation
    elif file_extension == '.txt':
        try:
            # Try to decode as UTF-8
            file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try other common encodings
            try:
                file_content.decode('latin-1')
            except UnicodeDecodeError:
                return False, "Invalid text file encoding"
    
    return True, f"Valid file type: {file_extension}"

def scan_for_malicious_content(file_content: bytes) -> Tuple[bool, str]:
    """
    Scan file content for known malicious signatures.
    Returns (is_safe, reason)
    """
    # Check for known malicious signatures
    for signature in MALICIOUS_SIGNATURES:
        if signature in file_content[:1024]:  # Check first 1KB
            return False, "Malicious signature detected"
    
    # Check for suspicious strings (basic heuristic)
    suspicious_strings = [
        b'cmd.exe', b'powershell', b'WScript.Shell',
        b'CreateObject', b'eval(', b'document.write',
        b'<script>', b'javascript:', b'vbscript:'
    ]
    
    content_lower = file_content[:10240].lower()  # Check first 10KB
    for suspicious in suspicious_strings:
        if suspicious.lower() in content_lower:
            return False, f"Suspicious content detected: {suspicious.decode('utf-8', errors='ignore')}"
    
    return True, "No malicious content detected"

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()

def quarantine_file(file_content: bytes, filename: str, reason: str) -> str:
    """
    Move suspicious file to quarantine directory.
    Returns quarantine file path.
    """
    quarantine_dir = setup_quarantine_directory()
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = calculate_file_hash(file_content)[:8]
    quarantine_filename = f"{timestamp}_{file_hash}_{filename}"
    quarantine_path = quarantine_dir / quarantine_filename
    
    # Save file to quarantine
    with open(quarantine_path, 'wb') as f:
        f.write(file_content)
    
    # Create metadata file
    metadata_path = quarantine_path.with_suffix('.meta')
    with open(metadata_path, 'w') as f:
        f.write(f"Original filename: {filename}\n")
        f.write(f"Quarantine reason: {reason}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"File hash: {file_hash}\n")
    
    return str(quarantine_path)

async def comprehensive_file_validation(file: UploadFile) -> Tuple[bool, str, Optional[bytes]]:
    """
    Perform comprehensive file validation.
    Returns (is_safe, message, file_content)
    """
    try:
        # Validate filename
        is_safe, message = validate_filename(file.filename)
        if not is_safe:
            return False, message, None
        
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return False, f"File extension not allowed: {file_extension}", None
        
        # Read file content
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        # Validate file size
        if len(file_content) > MAX_FILE_SIZE:
            return False, f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB", None
        
        if len(file_content) == 0:
            return False, "Empty file", None
        
        # Basic file type validation
        is_safe, message = validate_basic_file_type(file_content, file.filename)
        if not is_safe:
            return False, message, file_content
        
        # Scan for malicious content
        is_safe, message = scan_for_malicious_content(file_content)
        if not is_safe:
            return False, message, file_content
        
        return True, "File validation passed", file_content
        
    except Exception as e:
        return False, f"Validation error: {str(e)}", None
