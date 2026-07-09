# backend/orchestrator.py - Fix the ask method
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
import hashlib
import json

from backend.config import Config
from backend.document_processor import DocumentProcessor
from backend.chroma_store import ChromaStoreManager
from backend.qa_chain import QaChain
from backend.cache_manager import CacheManager
from backend.feedback_system import FeedbackSystem


class DocumentQaOrchestrator:
    def __init__(self, config=None):
        self.config = config or Config()
        
        print("\n" + "="*70)
        print("🚀 Starting Enhanced Document QA System")
        print("   Powered by Gemini 3.5 Flash")
        print("   ✦ LLM: Gemini 3.5 Flash")
        print("   ✦ Embeddings: gemini-embedding-001")
        print("   ✦ Features: Multi-Doc, Caching, Feedback")
        print("="*70)
        
        self.processor = DocumentProcessor(
            self.config.CHUNK_SIZE,
            self.config.CHUNK_OVERLAP
        )
        
        self.store = ChromaStoreManager(
            self.config.CHROMA_PERSIST_DIR,
            self.config.CHROMA_COLLECTION_NAME,
            self.config.EMBEDDING_MODEL
        )
        
        self.qa = QaChain(
            self.config.LLM_MODEL,
            self.config.TEMPERATURE
        )
        
        # Enhanced components
        self.cache = CacheManager()
        self.feedback = FeedbackSystem()
        
        # Document management
        self.ingested = set(self.store.get_all_sources())
        self.document_metadata = {}
        self.active_collection = self.config.CHROMA_COLLECTION_NAME
        
        # Performance tracking
        self.performance_stats = {
            'total_ingestions': len(self.ingested),
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_ingestion_time': 0,
            'avg_query_time': 0,
            'start_time': time.time()
        }
        
        print(f"\n✅ System Ready!")
        print(f"   📚 Documents ingested: {len(self.ingested)}")
        print(f"   🔮 LLM: {self.config.LLM_MODEL}")
        print(f"   🔢 Embeddings: {self.config.EMBEDDING_MODEL}")
        print(f"   ⚡ Cache: {'Enabled' if self.cache.enabled else 'Disabled'}")
        print("="*70 + "\n")
    
    def ingest_document(self, file_path: str, collection_name: Optional[str] = None) -> Dict:
        """Ingest a document into the system"""
        start_time = time.time()
        name = Path(file_path).name
        
        print(f"\n📥 Ingesting: {name}")
        
        if name in self.ingested:
            return {
                "success": True, 
                "message": f"Already ingested: {name}", 
                "already_ingested": True
            }
        
        if not Path(file_path).exists():
            return {"success": False, "message": f"File not found: {file_path}"}
        
        try:
            chunks = self.processor.process_file(file_path)
            
            if not chunks:
                return {"success": False, "message": "No content extracted from document"}
            
            # Use existing store
            self.store.add_documents(chunks)
            
            # Store metadata
            file_hash = self._get_file_hash(file_path)
            doc_metadata = {
                'file_name': name,
                'file_path': file_path,
                'file_size': Path(file_path).stat().st_size,
                'chunk_count': len(chunks),
                'collection': self.config.CHROMA_COLLECTION_NAME,
                'ingested_at': time.time(),
                'file_hash': file_hash
            }
            self.document_metadata[file_hash] = doc_metadata
            self.ingested.add(name)
            
            elapsed = time.time() - start_time
            self.performance_stats['total_ingestions'] += 1
            self.performance_stats['avg_ingestion_time'] = (
                (self.performance_stats['avg_ingestion_time'] * 
                 (self.performance_stats['total_ingestions'] - 1) + elapsed) / 
                self.performance_stats['total_ingestions']
            )
            
            return {
                "success": True,
                "message": f"✅ Successfully ingested: {name}",
                "chunks_created": len(chunks),
                "filename": name,
                "ingestion_time": f"{elapsed:.2f}s"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Failed: {str(e)}", "error": str(e)}
    
    def ask(self, question: str, collection_name: Optional[str] = None) -> Dict:
        """
        Ask a question to the document - FIXED VERSION
        """
        start_time = time.time()
        print(f"\n❓ Question: {question}")
        
        # Check if system has documents
        if not self.store.is_initialized and not self.ingested:
            return {
                "answer": "No documents ingested yet. Please upload a document first.",
                "sources": [],
                "confidence": "none",
                "error": "no_documents"
            }
        
        # Determine which collection to use
        target_collection = collection_name or self.active_collection
        
        # Check cache
        cache_key = self._get_cache_key(question, target_collection)
        cached_answer = self.cache.get(cache_key, target_collection)
        
        if cached_answer:
            self.performance_stats['cache_hits'] += 1
            cached_answer['from_cache'] = True
            cached_answer['cache_source'] = 'Redis'
            print(f"   ⚡ Answer from cache")
            return cached_answer
        
        self.performance_stats['cache_misses'] += 1
        
        try:
            # Use semantic search (simplified to avoid errors)
            docs = self.store.similarity_search(question, k=self.config.TOP_K_RESULTS)
            
            if not docs:
                return {
                    "answer": "I couldn't find any relevant information in the document to answer your question.",
                    "sources": [],
                    "confidence": "none",
                    "from_cache": False
                }
            
            # Generate answer using QA chain
            result = self.qa.generate_answer(question, docs)
            
            # Add metadata
            result['from_cache'] = False
            result['collection'] = target_collection
            result['query_time'] = time.time() - start_time
            result['search_type'] = 'semantic'
            
            # Cache the answer
            self.cache.set(cache_key, result, target_collection)
            
            # Update stats
            self.performance_stats['total_queries'] += 1
            self.performance_stats['avg_query_time'] = (
                (self.performance_stats['avg_query_time'] * 
                 (self.performance_stats['total_queries'] - 1) + result['query_time']) / 
                self.performance_stats['total_queries']
            )
            
            return result
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "confidence": "low",
                "error": str(e),
                "from_cache": False
            }
    
    def get_status(self) -> Dict:
        """Get system status"""
        feedback_stats = self.feedback.get_stats()
        cache_stats = self.cache.get_stats()
        
        return {
            "initialized": True,
            "total_files": len(self.ingested),
            "files": list(self.ingested),
            "store_initialized": self.store.is_initialized,
            "llm_model": self.config.LLM_MODEL,
            "embedding_model": self.config.EMBEDDING_MODEL,
            "chunk_size": self.config.CHUNK_SIZE,
            "top_k": self.config.TOP_K_RESULTS,
            "cache_enabled": self.cache.enabled,
            "performance": self.performance_stats,
            "feedback": {
                "total": feedback_stats.get('total_feedback', 0),
                "avg_rating": feedback_stats.get('average_rating', 0)
            },
            "cache_stats": cache_stats,
            "uptime": f"{time.time() - self.performance_stats['start_time']:.1f}s"
        }
    
    def clear_all(self):
        """Clear all documents from the system"""
        self.store.clear_collection()
        self.ingested.clear()
        self.document_metadata.clear()
        self.cache.clear_cache()
        self.performance_stats['total_ingestions'] = 0
        print("🗑️ All documents cleared from the system")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate file hash for identification"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_cache_key(self, question: str, collection: str) -> str:
        """Generate cache key"""
        text = f"{question}_{collection}"
        return hashlib.md5(text.encode()).hexdigest()
    
    def add_feedback(self, question: str, answer: str, rating: int, comments: str = ""):
        """Add user feedback"""
        return self.feedback.add_feedback(question, answer, 'user_feedback', rating, comments)
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        return self.feedback.get_stats()
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get improvement suggestions from feedback"""
        return self.feedback.get_improvement_suggestions()
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear_cache()
        print("✅ Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def export_session(self, filename: str = "session_export.json") -> Dict:
        """Export session data to JSON"""
        export_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'llm_model': self.config.LLM_MODEL,
                'embedding_model': self.config.EMBEDDING_MODEL,
                'chunk_size': self.config.CHUNK_SIZE,
                'top_k': self.config.TOP_K_RESULTS
            },
            'documents': self.document_metadata,
            'performance': self.performance_stats,
            'feedback': self.feedback.feedback_data
        }
        
        os.makedirs('storage/exports', exist_ok=True)
        filepath = f"storage/exports/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Session exported to {filepath}")
        return export_data