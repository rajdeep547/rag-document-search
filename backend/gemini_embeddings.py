# backend/gemini_embeddings.py
"""
Gemini Embeddings using new google-genai SDK
"""
import os
from typing import List
from dotenv import load_dotenv
from google import genai

load_dotenv()

class GeminiEmbeddings:
    """
    Gemini Embeddings wrapper using the new google-genai SDK
    """
    
    def __init__(self, model="gemini-embedding-001"):
        """
        Initialize Gemini embeddings
        
        Args:
            model: Embedding model name (default: gemini-embedding-001)
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        # Initialize the new client
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self._dimension = None
        print(f"✅ GeminiEmbeddings ready with {self.model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            embeddings = []
            for text in texts:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=text
                )
                embeddings.append(result.embeddings[0].values)
            return embeddings
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            # Return dummy embeddings if API fails
            return [[0.0] * 768 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"❌ Query embedding error: {e}")
            return [0.0] * 768
    
    def __call__(self, text):
        """
        Allow the object to be called directly
        
        Args:
            text: String or list of strings
            
        Returns:
            Embedding(s)
        """
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension"""
        if self._dimension is None:
            try:
                # Get a sample embedding to determine dimension
                sample = self.embed_query("test")
                self._dimension = len(sample)
            except:
                self._dimension = 768  # Default fallback
        return self._dimension