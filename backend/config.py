# backend/config.py
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    # Google Gemini Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # ✅ USE GEMINI 3.5 FLASH (NOT 1.5)
    LLM_MODEL = "gemini-3.5-flash"  # ← Make sure this is correct
    
    # Embedding model (working)
    EMBEDDING_MODEL = "gemini-embedding-001"
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval
    TOP_K_RESULTS = 3
    
    # Timeout
    REQUEST_TIMEOUT = 55
    
    # Chroma Configuration
    CHROMA_PERSIST_DIR = "./storage/chroma_db"
    CHROMA_COLLECTION_NAME = "document_collection"
    
    # Temperature
    TEMPERATURE = 0
    
    # Supported file types
    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]

if not Config.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

print("✅ Config loaded (Gemini 3.5 Flash)")
print(f"   LLM Model: {Config.LLM_MODEL}")
print(f"   Embedding Model: {Config.EMBEDDING_MODEL}")