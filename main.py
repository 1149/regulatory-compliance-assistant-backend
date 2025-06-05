from fastapi import FastAPI, Depends 
from sqlalchemy.orm import Session 
from database import SessionLocal, engine, Base, Document

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get a DB session. This will be used in the API routes.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI Backend!"}

# Adding a test route to ensure the DB connection works
@app.get("/test-db")
async def test_db_connection(db: Session = Depends(get_db)):
    try:
        # Trying to execute a simple query to check connection
        db.query(Document).first()
        return {"message": "Database connection successful!"}
    except Exception as e:
        return {"message": f"Database connection failed: {e}"}
