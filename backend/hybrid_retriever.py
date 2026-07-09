# backend/hybrid_retriever.py
"""
Hybrid Search Implementation
Combines semantic search (vectors) with keyword search (BM25)
"""
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple, Optional, Dict
import re
import math


class HybridRetriever:
    """
    Hybrid retriever combining BM25 (keyword) and semantic (vector) search
    """
    
    def __init__(self, vector_store, bm25_weight: float = 0.3):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: ChromaDB vector store
            bm25_weight: Weight for BM25 (0.0 = pure semantic, 1.0 = pure keyword)
        """
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.documents = []
        self.doc_ids = []
        self.bm25_index = None
        self.is_indexed = False
        
    def index_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """
        Index documents for BM25 search
        
        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
        """
        if not documents:
            return
        
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.is_indexed = True
        
        print(f"✅ Hybrid retriever indexed {len(documents)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split by non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        
        # Remove stopwords (optional - basic set)
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 
                     'at', 'to', 'by', 'in', 'of', 'with', 'without', 'is', 'are',
                     'am', 'was', 'were', 'be', 'been', 'being'}
        
        return [t for t in tokens if t not in stopwords and len(t) > 1]
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Perform hybrid search combining BM25 and semantic search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        if not self.is_indexed or not self.documents:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # 1. BM25 (Keyword) Search
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0, 1]
        if len(bm25_scores) > 0:
            bm25_scores = self._normalize_scores(bm25_scores)
        
        # 2. Semantic Search (Vector)
        try:
            semantic_results = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            # Extract semantic scores
            semantic_scores = {}
            for doc, score in semantic_results:
                # Try to find document index
                idx = self._find_doc_index(doc)
                if idx is not None:
                    semantic_scores[idx] = score
        except Exception as e:
            print(f"⚠️ Semantic search failed: {e}")
            semantic_scores = {}
        
        # 3. Combine scores using Reciprocal Rank Fusion (RRF)
        combined_scores = {}
        
        # Add BM25 scores with weight
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                # Convert to rank-based score using RRF
                rank = np.argsort(bm25_scores)[::-1].tolist().index(idx)
                rrf_score = 1 / (rank + 60)
                combined_scores[idx] = combined_scores.get(idx, 0) + (1 - self.bm25_weight) * rrf_score
        
        # Add semantic scores with weight
        for idx, score in semantic_scores.items():
            # Convert similarity to rank-based score
            rank = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
            rank = [r[0] for r in rank].index(idx)
            rrf_score = 1 / (rank + 60)
            combined_scores[idx] = combined_scores.get(idx, 0) + self.bm25_weight * rrf_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return sorted_results[:k]
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _find_doc_index(self, doc) -> Optional[int]:
        """
        Find the index of a document in the stored list
        """
        # Try to find by content
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        for idx, stored_doc in enumerate(self.documents):
            if content == stored_doc or content in stored_doc:
                return idx
        
        # Try to find by metadata
        if hasattr(doc, 'metadata') and 'index' in doc.metadata:
            return doc.metadata['index']
        
        return None
    
    def search_with_weights(self, query: str, k: int = 5, semantic_weight: float = 0.7) -> List[Tuple[int, float]]:
        """
        Search with custom weights for semantic vs keyword
        
        Args:
            query: Search query
            k: Number of results
            semantic_weight: Weight for semantic search (0.0 to 1.0)
            
        Returns:
            List of (document_index, score) tuples
        """
        # Temporarily override BM25 weight
        original_weight = self.bm25_weight
        self.bm25_weight = 1.0 - semantic_weight
        
        results = self.hybrid_search(query, k)
        
        # Restore original weight
        self.bm25_weight = original_weight
        
        return results


class RRFHybridRetriever(HybridRetriever):
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF) with additional features
    """
    
    def __init__(self, vector_store, bm25_weight: float = 0.3, k: int = 60):
        super().__init__(vector_store, bm25_weight)
        self.rrf_k = k  # RRF constant
        
    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Enhanced hybrid search with RRF
        """
        if not self.is_indexed or not self.documents:
            return []
        
        tokenized_query = self._tokenize(query)
        
        # 1. BM25 Search
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        # 2. Semantic Search
        try:
            semantic_results = self.vector_store.similarity_search_with_score(query, k=min(k*2, len(self.documents)))
            semantic_ranks = []
            for doc, score in semantic_results:
                idx = self._find_doc_index(doc)
                if idx is not None:
                    semantic_ranks.append((idx, score))
            semantic_ranks.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"⚠️ Semantic search failed: {e}")
            semantic_ranks = []
        
        # 3. Combine using RRF
        combined_scores = {}
        
        # Add BM25 scores
        for rank, idx in enumerate(bm25_ranks):
            if rank < len(bm25_scores):
                rrf_score = 1 / (rank + self.rrf_k)
                combined_scores[idx] = combined_scores.get(idx, 0) + (1 - self.bm25_weight) * rrf_score
        
        # Add semantic scores
        for rank, (idx, score) in enumerate(semantic_ranks):
            rrf_score = 1 / (rank + self.rrf_k)
            combined_scores[idx] = combined_scores.get(idx, 0) + self.bm25_weight * rrf_score
        
        # Sort and return top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]