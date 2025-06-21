# SpaCy and NLP utility functions
import spacy
import google.generativeai as genai
from config import GEMINI_MODEL_NAME

# --- SpaCy Initialization ---
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading SpaCy model 'en_core_web_md'...")
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

def extract_entities_with_spacy(text: str):
    """Extract entities using SpaCy NER model."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })
    return entities

def post_process_spacy_entities(entities: list):
    """Post-process SpaCy entities for compliance domain using Gemini."""
    processed_entities = []
    
    for ent in entities:
        text = ent["text"]
        label = ent["label"]
        
        print(f"DEBUG_POST: Processing entity: '{text[:80]}...' (Initial Label: {label})")
        original_label = label

        # Determine if we should reclassify this entity - now broader criteria
        should_reclassify = False
        if label in ["ORG", "MISC", "LAW", "FAC", "GPE", "PERSON", "DATE", "TIME", "CARDINAL", "ORDINAL"] and (
            len(text.split()) <= 5 or  # Short entities are often misclassified
            any(keyword in text.lower() for keyword in [
                "section", "policy", "control", "procedure", "requirement", "article", "clause",
                "regulation", "standard", "framework", "compliance", "audit", "review", "assessment",
                "inc", "corp", "ltd", "llc", "company", "organization", "agency", "department",
                "january", "february", "march", "april", "may", "june", "july", "august", 
                "september", "october", "november", "december", "monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday", "sunday", "am", "pm", "o'clock", "hours", "minutes"
            ]) or
            text.upper() in ["GDPR", "CCPA", "HIPAA", "SOX", "MFA", "ISO", "NIST", "PCI", "DSS"]
        ):
            should_reclassify = True

        if should_reclassify:
            try:
                gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                reclassify_prompt = f"""You are an expert entity classifier. Analyze the text and classify it into the most appropriate category.

ENTITY CLASSIFICATION TASK:
Classify this text: "{text}"

AVAILABLE CATEGORIES:
• LAW - Legal regulations, statutes, acts, compliance frameworks, standards, policy sections
  Examples: "GDPR Article 6", "Section 3.2", "ISO 27001", "Privacy Act", "SOX requirements"

• ORGANIZATION - Companies, institutions, agencies, departments, groups
  Examples: "Microsoft Corporation", "FDA", "IT Department", "European Commission", "Third parties"

• PERSON - Individual names, job titles, roles, positions
  Examples: "John Smith", "Data Protection Officer", "Chief Security Officer", "employees", "users"

• DATE - Specific dates, time periods, deadlines, schedules
  Examples: "January 2024", "within 30 days", "quarterly", "annually", "2023-12-31"

• TIME - Specific times, durations, frequencies
  Examples: "9:00 AM", "2 hours", "real-time", "daily", "immediately", "24/7"

• LOCATION - Places, regions, countries, addresses, jurisdictions
  Examples: "United States", "European Union", "headquarters", "cloud servers", "data centers"

• MONEY - Financial amounts, costs, fees, monetary values
  Examples: "$1,000", "500 euros", "processing fees", "budget allocation", "costs"

• PERCENT - Percentages, ratios, proportions
  Examples: "99.9%", "two-thirds", "majority", "50 percent", "half"

• CARDINAL - Numbers, quantities, amounts, counts
  Examples: "5 million records", "three instances", "100 users", "first time", "256-bit"

• ORDINAL - Order, sequence, ranking, priority
  Examples: "first", "second review", "third party", "primary", "initial assessment"

• MISC - Other important entities, technical terms, concepts not fitting above categories
  Examples: "encryption", "data breach", "access control", "authentication", "backup systems"

INSTRUCTIONS:
1. Choose the MOST APPROPRIATE single category
2. Consider the context and meaning of the text
3. Be consistent in classification
4. Respond with ONLY the category name (e.g., "LAW")

CLASSIFICATION:"""
                response = gemini_model.generate_content(
                    reclassify_prompt,
                    generation_config=genai.types.GenerationConfig(max_output_tokens=10)
                )
                
                new_label_raw = response.text.strip().upper()
                if new_label_raw in ["LAW", "ORGANIZATION", "PERSON", "DATE", "TIME", "LOCATION", "MONEY", "PERCENT", "CARDINAL", "ORDINAL", "MISC"]:
                    if new_label_raw == "ORGANIZATION":
                        new_label = "ORG"
                    elif new_label_raw == "LOCATION":
                        new_label = "GPE"
                    else:
                        new_label = new_label_raw

                    label = new_label
                    print(f"DEBUG_POST:   LLM changed label from {original_label} to {label} for '{text[:80]}...'")
                else:
                    print(f"DEBUG_POST:   LLM returned unmappable label '{new_label_raw}'. Label remains {original_label} for '{text[:80]}...'")

            except Exception as e:
                print(f"Warning: Failed to reclassify entity '{text[:80]}...' with Gemini: {e}")

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
