# backend/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # ========================================================================
    # API Configuration
    # ========================================================================
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # ========================================================================
    # Model Configuration - Using correct model names
    # ========================================================================
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")  # Correct name
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # ========================================================================
    # Document Processing
    # ========================================================================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    CHUNK_SEPARATORS = os.getenv("CHUNK_SEPARATORS", "\n\n").split(",")
    
    # ========================================================================
    # Retrieval Configuration
    # ========================================================================
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    HYBRID_SEARCH_WEIGHT = float(os.getenv("HYBRID_SEARCH_WEIGHT", "0.3"))
    
    # ========================================================================
    # Storage Configuration
    # ========================================================================
    BASE_DIR = Path(__file__).parent.parent
    STORAGE_DIR = BASE_DIR / "storage"
    
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(STORAGE_DIR / "chroma_db"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "document_collection")
    FEEDBACK_STORAGE = os.getenv("FEEDBACK_STORAGE", str(STORAGE_DIR / "feedback.json"))
    SESSION_EXPORT_DIR = os.getenv("SESSION_EXPORT_DIR", str(STORAGE_DIR / "exports"))
    
    # ========================================================================
    # Cache Configuration
    # ========================================================================
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    
    # ========================================================================
    # Performance Configuration
    # ========================================================================
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "55"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
    
    # ========================================================================
    # Supported File Types
    # ========================================================================
    SUPPORTED_EXTENSIONS = os.getenv("SUPPORTED_EXTENSIONS", ".pdf,.txt,.md,.docx,.csv").split(",")
    
    # ========================================================================
    # Validation
    # ========================================================================
    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        # Create storage directory
        cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        return True

# Validate on import
Config.validate()
print("✅ Config loaded (Gemini 1.5 Flash)")
print(f"   📊 LLM Model: {Config.LLM_MODEL}")
print(f"   🔢 Embedding Model: {Config.EMBEDDING_MODEL}")