# Regulatory Compliance Assistant Backend - Main Application
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Import configuration and database
from config import CORS_ORIGINS
from database import SessionLocal, engine, Base, Document

# Import route modules
from routes.document_routes import router as document_router
from routes.upload_routes import router as upload_router
from routes.ai_routes import router as ai_router

# Create all database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Regulatory Compliance Assistant API",
    description="Backend API for document processing, entity extraction, and compliance analysis",
    version="1.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Include route modules
app.include_router(document_router)
app.include_router(upload_router)
app.include_router(ai_router)

# Root endpoints
@app.get("/")
async def read_root():
    return {
        "message": "Regulatory Compliance Assistant API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "regulatory-compliance-backend"}

@app.get("/test-db")
async def test_db_connection(db: Session = Depends(get_db)):
    """Test database connection."""
    try:
        db.query(Document).first()
        return {"message": "Database connection successful!"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection failed: {e}"
        )
