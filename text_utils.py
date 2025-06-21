# Text processing and RAG utilities
import re
import unicodedata
from nlp_utils import nlp
from config import CHUNK_SIZE, CHUNK_OVERLAP

def clean_extracted_text(raw_text: str) -> str:
    """Clean and normalize text extracted from PDF."""
    extracted_text = raw_text

    # Replace sequences of two or more newlines with double newline
    extracted_text = re.sub(r'\n\s*\n', '\n\n', extracted_text)

    # Replace single newlines within sentences with space
    extracted_text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', extracted_text)

    # Replace multiple spaces with single space
    extracted_text = re.sub(r' +', ' ', extracted_text)

    # Strip leading/trailing whitespace
    extracted_text = extracted_text.strip()

    # Normalize Unicode characters
    extracted_text = unicodedata.normalize("NFKC", extracted_text)
    
    # Replace all types of newlines with space
    extracted_text = extracted_text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    
    # Replace any sequence of whitespace with single space
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()

    return extracted_text

def get_text_chunks(text: str):
    """
    Splits text into overlapping chunks using a sliding window approach
    to ensure context is not lost at chunk boundaries.
    """
    if not text:
        return []

    # Use sentence-aware processing
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
        
        # Next window starts with overlap
        next_start = start_index + CHUNK_SIZE - CHUNK_OVERLAP
        
        # Prevent infinite loop
        if next_start <= start_index:
            start_index += 1
        else:
            start_index = next_start
    
    return chunks

def format_document_display(text: str, document_title: str = None, document_id: int = None) -> str:
    """
    Format document text for clean, professional, and readable display.
    """
    if not text or not text.strip():
        return "Document appears to be empty or contains no readable text."
    
    import textwrap
    import re
    
    # Clean the text but preserve structure
    text = ' '.join(text.split())
    
    result = []
    
    # Clean header
    result.append("=" * 80)
    if document_title:
        result.append(f"  📄 {document_title}")
    elif document_id:
        result.append(f"  📄 Document ID: {document_id}")
    else:
        result.append("  📄 Document Content")
    result.append("=" * 80)
    result.append("")
    
    # Look for sections with improved pattern
    section_splits = re.split(r'(Section\s+\d+(?:\.\d+)*(?:\s*[-:]\s*|\s+))', text, flags=re.IGNORECASE)
    
    # Clean empty parts
    parts = [part.strip() for part in section_splits if part.strip()]
    
    if len(parts) > 1:
        # Process sections
        i = 0
        while i < len(parts):
            part = parts[i]
            
            if re.match(r'Section\s+\d+', part, re.IGNORECASE):
                # This is a section header
                section_title = part.strip()
                
                # Get content for this section
                content = ""
                if i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if not re.match(r'Section\s+\d+', next_part, re.IGNORECASE):
                        content = next_part.strip()
                        i += 1  # Skip the content part as we've used it
                
                # Add section with clean formatting
                result.append(f"🔹 {section_title.title()}")
                result.append("   " + "-" * 60)
                
                if content:
                    # Break content into readable paragraphs
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    para_sentences = []
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                        
                        para_sentences.append(sentence.strip())
                        
                        # Create paragraph every 2-3 sentences or when getting long
                        if len(para_sentences) >= 3 or len(' '.join(para_sentences)) > 300:
                            para_text = ' '.join(para_sentences)
                            
                            # Use textwrap properly for clean wrapping
                            lines = textwrap.wrap(para_text, width=72, 
                                                break_long_words=False, 
                                                break_on_hyphens=False)
                            
                            for line in lines:
                                result.append(f"   {line}")
                            result.append("")
                            
                            para_sentences = []
                    
                    # Add any remaining sentences
                    if para_sentences:
                        para_text = ' '.join(para_sentences)
                        lines = textwrap.wrap(para_text, width=72, 
                                            break_long_words=False, 
                                            break_on_hyphens=False)
                        for line in lines:
                            result.append(f"   {line}")
                        result.append("")
                
                result.append("")  # Extra space after section
                i += 1
            else:
                # Regular content
                lines = textwrap.wrap(part, width=75, 
                                    break_long_words=False, 
                                    break_on_hyphens=False)
                result.extend(lines)
                result.append("")
                i += 1
    else:
        # No sections - format as simple paragraphs
        sentences = re.split(r'(?<=[.!?])\s+', text)
        para_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            para_sentences.append(sentence.strip())
            
            # Create paragraphs
            if len(para_sentences) >= 3 or len(' '.join(para_sentences)) > 250:
                para_text = ' '.join(para_sentences)
                lines = textwrap.wrap(para_text, width=75, 
                                    break_long_words=False, 
                                    break_on_hyphens=False)
                result.extend(lines)
                result.append("")
                para_sentences = []
        
        # Add remaining content
        if para_sentences:
            para_text = ' '.join(para_sentences)
            lines = textwrap.wrap(para_text, width=75, 
                                break_long_words=False, 
                                break_on_hyphens=False)
            result.extend(lines)
            result.append("")
    
    # Simple footer
    word_count = len(text.split())
    char_count = len(text)
    
    result.append("=" * 80)
    result.append(f"  📊 Document Statistics: {word_count} words | {char_count} characters")
    result.append("=" * 80)
    
    return '\n'.join(result)

def wrap_text(text: str, width: int = 75) -> list:
    """
    Wrap text to specified width, breaking at word boundaries only.
    """
    if not text:
        return []
    
    import textwrap
    # Use textwrap for proper word wrapping that doesn't break words
    wrapped_lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return wrapped_lines


