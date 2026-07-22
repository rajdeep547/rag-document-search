# backend/local_embeddings.py
"""
Local embeddings using sentence-transformers (no API key required)
"""
from typing import List
from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    """
    Local embeddings wrapper using sentence-transformers
    """

    def __init__(self, model="all-MiniLM-L6-v2"):
        """
        Initialize local embeddings

        Args:
            model: sentence-transformers model name (default: all-MiniLM-L6-v2)
        """
        self.model_name = model
        self.model = SentenceTransformer(model)
        self._dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ LocalEmbeddings ready with {self.model_name}")

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
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return [[0.0] * self._dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Query embedding error: {e}")
            return [0.0] * self._dimension

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
        return self._dimension
