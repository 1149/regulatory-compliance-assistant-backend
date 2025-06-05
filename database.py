from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os # Import the os module
from dotenv import load_dotenv # Import load_dotenv

load_dotenv()
# --- Database Configuration ---
# IMPORTANT: Replace 'your_compliance_user_password' with the actual password you set for 'compliance_user'
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
    status = Column(String, default="pending_processing") # e.g., 'pending', 'processed', 'error'
    path_to_text = Column(String, nullable=True) # Path to where the extracted text file will be stored

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"