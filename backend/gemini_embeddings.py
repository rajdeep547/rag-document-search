# backend/gemini_embeddings.py
"""
Gemini 3.5 Flash Embeddings - Using Google's Gemini embedding API
"""

from typing import List
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbeddings:
    def __init__(self, model="gemini-embedding-001", api_key=None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Set it in .env file")
        
        # Configure with API key
        genai.configure(api_key=self.api_key)
        
        # Get embedding dimension by testing
        try:
            test_result = genai.embed_content(
                model=self.model,
                content="test",
                task_type="retrieval_document"
            )
            self.dimension = len(test_result['embedding'])
            print(f"✅ Gemini Embeddings initialized")
            print(f"   Model: {model}")
            print(f"   Dimension: {self.dimension}")
        except Exception as e:
            print(f"⚠️ Could not determine dimension: {e}")
            self.dimension = 3072
            print(f"   Assuming dimension: {self.dimension}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query for search"""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents for storage"""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings