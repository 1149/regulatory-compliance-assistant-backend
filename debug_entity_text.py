# debug_entity_text.py
import re

# Get the exact problematic text from your last FastAPI terminal output
# It was: 'Section 3.0 - Access Control 3.1 Principle of Least Privilege: Access'
# Copy it precisely, including quotes from the DEBUG: SpaCy Extracted Entities (Post-Processed) list.
problematic_text = 'Section 3.0 - Access Control 3.1 Principle of Least Privilege: Access'

print(f"--- Debugging String '{problematic_text[:80]}...' ---")
print(f"repr(problematic_text): '{repr(problematic_text)}'")
print(f"Length of text: {len(problematic_text)}")

# Check for hidden characters (e.g., non-breaking spaces)
print("Character breakdown (char, ASCII/Unicode ordinal):")
for i, char in enumerate(problematic_text):
    print(f"  [{i}]: '{char}' (ord: {ord(char)})")

# Test the specific conditions
access_control_check = "Access Control" in problematic_text
section_check = "Section" in problematic_text
combined_check = access_control_check and section_check

print(f"\n'Access Control' in text: {access_control_check}")
print(f"'Section' in text: {section_check}")
print(f"Combined condition ('Access Control' in text and 'Section' in text): {combined_check}")

# Test the regex rule as well
regex_match = re.match(r'^(Section|Policy)\s+(\d+(\.\d+)*)?\s*[-:]?\s*.*', problematic_text, re.IGNORECASE)
print(f"Regex match result: {bool(regex_match)}")
if regex_match:
    print(f"Regex match groups: {regex_match.groups()}")

print("--- End Debugging ---")