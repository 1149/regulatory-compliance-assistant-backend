from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text 
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import json 

load_dotenv()

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please create a .env file with it.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Model Definition ---
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending_processing")  # e.g., 'pending', 'processed', 'error'
    path_to_text = Column(String, nullable=True)  # Path to where the extracted text file will be stored
    embeddings = Column(Text, nullable=True)  # NEW: Column to store embeddings as TEXT (JSON string)

    # Relationship to Entity
    entities = relationship("Entity", back_populates="document")

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"

class Entity(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True)  # Link to Document
    text = Column(String)  # The actual entity text (e.g., "GDPR")
    label = Column(String)  # The type of entity (e.g., "ORG", "DATE", "LAW")
    start_char = Column(Integer)  # Starting character index in the original text
    end_char = Column(Integer)  # Ending character index in the original text

    # Define relationship to Document
    document = relationship("Document", back_populates="entities")

    def __repr__(self):
        return f"<Entity(id={self.id}, doc_id={self.document_id}, text='{self.text}', label='{self.label}')>"

class DocumentSchema(BaseModel):
    id: int
    filename: str
    upload_date: datetime  # FastAPI/Pydantic can handle datetime
    status: str
    path_to_text: Optional[str] = None  # Optional because it can be nullable in DB
    embeddings: Optional[str] = None  # NEW: Add embeddings to schema

    class Config:
        from_attributes = True

class EntitySchema(BaseModel):
    id: int
    document_id: int
    text: str
    label: str
    start_char: int
    end_char: int

    class Config:
        from_attributes = True