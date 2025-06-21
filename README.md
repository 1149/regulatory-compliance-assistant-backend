# 🏛️ Regulatory Compliance Assistant Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-00a393?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![AI Powered](https://img.shields.io/badge/AI-Powered-ff6b6b?style=flat&logo=openai)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An intelligent FastAPI backend for automated document processing, compliance analysis, and regulatory insights powered by AI.**

##  Overview

The **Regulatory Compliance Assistant** is a sophisticated backend system designed to help organizations manage, analyze, and ensure compliance with regulatory documents. Using cutting-edge AI technologies, it automatically processes PDF documents, extracts key entities, identifies compliance clauses, and provides intelligent Q&A capabilities.

###  **Key Features**

- ** Smart Document Processing** - Automatic PDF text extraction and cleaning
- ** AI-Powered Analysis** - Entity extraction and compliance clause identification  
- ** Intelligent Q&A** - RAG (Retrieval Augmented Generation) for document queries
- ** Advanced Search** - Semantic search across document collections
- ** Entity Recognition** - Automated extraction of people, organizations, dates, and compliance terms
- **🏗 Modular Architecture** - Clean, maintainable, and scalable codebase
- ** High Performance** - Optimized for handling large document volumes

##  AI Capabilities

### **Dual AI Engine Support**
- **🌐 Google Gemini AI** - Cloud-based AI for advanced analysis and embeddings
- **🏠 Local Ollama** - Privacy-focused local LLM for sensitive documents

### **Intelligent Features**
- **Entity Extraction**: Automatically identifies and categorizes:
  - 👥 People (executives, compliance officers, contacts)
  - 🏢 Organizations (companies, regulatory bodies, vendors)
  - 📅 Dates (effective dates, review periods, deadlines)
  - 📋 Compliance Terms (policies, procedures, requirements)
  - 🔢 Financial Data (amounts, percentages, metrics)

- **Compliance Analysis**: 
  - Identifies regulatory clauses and requirements
  - Extracts policy statements and procedures
  - Highlights compliance obligations and deadlines

- **RAG-Powered Q&A**:
  - Ask natural language questions about documents
  - Get contextual answers with source attribution
  - Intelligent document chunking for accurate responses

## 🏗️ Architecture

### **Clean, Modular Design**
```
regulatory-compliance-assistant-backend/
├──  main.py                    # FastAPI application entry point
├──  config.py                  # Configuration and environment settings
├──  database.py                # SQLAlchemy models and schemas
├──  requirements.txt           # Python dependencies
│
├── 🔧 Core AI & Processing/
│   ├──  ai_services.py         # Google Gemini AI integration
│   ├──  ollama_service.py      # Local Ollama LLM service
│   ├──  nlp_utils.py           # SpaCy NLP and entity processing
│   ├──  text_utils.py          # Text cleaning and document formatting
│   └──  file_utils.py          # File operations and management
│
├── 🛣️ API Routes/
│   ├──  document_routes.py     # Document management endpoints
│   ├──  upload_routes.py       # File upload and processing
│   └──  ai_routes.py           # AI analysis and Q&A endpoints
│
└── 📁 uploads/                   # Document storage
```

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- Google Gemini API key (optional, for cloud AI)
- Ollama installed (optional, for local AI)

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/regulatory-compliance-assistant-backend.git
   cd regulatory-compliance-assistant-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

4. **Run the server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the API**
   - API Documentation: `http://localhost:8000/docs`
   - Interactive UI: `http://localhost:8000/redoc`

## 📋 API Endpoints

### **📄 Document Management**
```http
GET    /api/documents/                    # List all documents
GET    /api/documents/subjects/           # Get unique document subjects
GET    /api/documents/{id}/entities       # Extract entities from document
GET    /api/documents/{id}/text           # Get beautifully formatted text
GET    /api/documents/{id}/raw-text       # Get raw document text
GET    /api/documents/{id}/pdf            # Download original PDF
DELETE /api/documents/{id}/               # Delete document
```

### **📤 Document Upload & Processing**
```http
POST   /api/upload-document/              # Upload PDF with automatic processing
```

### **🤖 AI Analysis & Intelligence**
```http
POST   /api/summarize-text/               # Summarize text with Gemini AI
POST   /api/summarize-text-local/         # Summarize with local Ollama
POST   /api/document/{id}/qa/             # Ask questions about documents
POST   /api/analyze-policy/               # Analyze policy compliance
POST   /api/test-ner/                     # Test named entity recognition
POST   /api/test-chunking/                # Test document chunking
```

## 🎨 Document Formatting

The API provides beautiful, professional document formatting:

```
================================================================================
  📄 Data Security and Incident Response Policy
================================================================================

COMPANY: TechSolutions Inc. POLICY ID: TS-DSP-001 VERSION: 2.1
EFFECTIVE DATE: September 1, 2024 REVIEW DATE: September 1, 2025

🔹 Section 1.0 - Purpose And Scope:
   ------------------------------------------------------------
   This policy establishes the framework for protecting TechSolutions
   Inc.'s information assets from unauthorized access, use, disclosure,
   disruption, modification, or destruction. It applies to all employees,
   contractors, and third parties with access to company data or systems.

🔹 Section 2.0 - Data Classification And Handling:
   ------------------------------------------------------------
   All data categorized as 'Confidential' must be encrypted both in transit
   and at rest. Access to confidential data is strictly on a need-to-know
   basis.

================================================================================
  📊 Document Statistics: 360 words | 2,512 characters
================================================================================
```

## 🔧 Configuration

### **Environment Variables**
```bash
# AI Services
GEMINI_API_KEY=your_gemini_api_key_here
OLLAMA_BASE_URL=http://localhost:11434

# Database
DATABASE_URL=sqlite:///./compliance.db

# Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE=10485760  # 10MB
```

### **Supported File Types**
- 📄 PDF documents
- 📝 Text files (for testing and analysis)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📊 Use Cases

### **Regulatory Compliance Teams**
- Automated compliance document analysis
- Quick identification of regulatory requirements
- Deadline and obligation tracking

### **Legal Departments**
- Contract and policy review automation
- Entity and relationship extraction
- Compliance risk assessment

### **Risk Management**
- Policy gap analysis
- Regulatory change impact assessment
- Automated compliance reporting

### **Audit Preparation**
- Document organization and categorization
- Quick access to relevant policy sections
- Evidence collection and presentation

## 🛡️ Security & Privacy

- **Local Processing Option**: Use Ollama for sensitive documents that can't leave your infrastructure
- **Data Encryption**: All uploaded documents are securely stored
- **API Security**: Built-in rate limiting and authentication ready
- **Privacy First**: No data sent to external services when using local AI mode

## 📈 Performance

- **Fast Processing**: Optimized PDF parsing and text extraction
- **Scalable Architecture**: Modular design supports horizontal scaling
- **Efficient AI**: Smart chunking and caching for optimal AI performance
- **Resource Management**: Configurable memory and processing limits

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint for interactive API documentation
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join the community discussions for questions and ideas

## 🙏 Acknowledgments

- **FastAPI** - For the excellent web framework
- **SpaCy** - For powerful NLP capabilities
- **Google Gemini** - For advanced AI analysis
- **Ollama** - For local LLM support

---

<div align="center">

**🌟 Star this repository if you find it helpful! 🌟**

Made with ❤️ for the compliance and regulatory community

</div>
