# backend/chroma_store.py
import os
import time
import shutil
from typing import List, Dict, Any, Optional, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from backend.gemini_embeddings import GeminiEmbeddings

class ChromaStoreManager:
    def __init__(self, 
                 persist_dir="./storage/chroma_db", 
                 collection="docs",
                 embedding_model="gemini-embedding-001"):
        self.persist_dir = persist_dir
        self.collection = collection
        self.embedding_model = embedding_model
        
        # Use Gemini for embeddings
        self.embeddings = GeminiEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.is_initialized = False
        self._init()
        print("✅ ChromaStore ready (Gemini embeddings)")
    
    def _init(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        db_path = os.path.join(self.persist_dir, "chroma.sqlite3")
        if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=self.collection
                )
                self.is_initialized = True
                print("   ✅ Loaded existing DB")
            except Exception as e:
                print(f"   📝 Will create new DB")
        else:
            print("   📝 New DB will be created")
    
    def add_documents(self, chunks: List[Document]) -> int:
        print(f"➕ Adding {len(chunks)} chunks...")
        if not self.is_initialized or self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=self.collection
            )
            self.is_initialized = True
        else:
            self.vectorstore.add_documents(chunks)
        print(f"   ✅ Added")
        return len(chunks)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.is_initialized or self.vectorstore is None:
            return []
        
        print(f"🔍 Searching...")
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            return []
        
        formatted = []
        for doc, score in results:
            similarity = 1 / (1 + score) if score > 0 else 1.0
            formatted.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": round(similarity, 3)
            })
        
        print(f"   ✅ Found {len(formatted)} chunks")
        if formatted:
            print(f"   📊 Top score: {formatted[0]['similarity_score']}")
        return formatted
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores - returns Document objects with scores
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.is_initialized or self.vectorstore is None:
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            return []
    
    def get_document_by_index(self, index: int) -> Optional[Document]:
        """
        Get a document by its index - NEW METHOD
        
        Args:
            index: Document index or ID
            
        Returns:
            Document object or None
        """
        if not self.is_initialized or self.vectorstore is None:
            return None
        
        try:
            # Get all documents
            results = self.vectorstore.get()
            
            if results and results['documents'] and results['metadatas']:
                # Check if we can find by metadata
                for i, (doc_text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    # Check by index in metadata or by position
                    if metadata and metadata.get('chunk_index') == index:
                        return Document(page_content=doc_text, metadata=metadata)
                    if i == index:
                        return Document(page_content=doc_text, metadata=metadata)
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting document: {e}")
            return None
    
    def get_all_sources(self) -> list:
        if not self.is_initialized or self.vectorstore is None:
            return []
        try:
            results = self.vectorstore.get()
            sources = set()
            for m in results['metadatas']:
                if m and 'source' in m:
                    sources.add(m['source'])
            return sorted(list(sources))
        except Exception:
            return []
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        if self.is_initialized and self.vectorstore:
            try:
                results = self.vectorstore.get()
                if results['ids']:
                    self.vectorstore.delete(ids=results['ids'])
                    print("🗑️ Collection cleared")
                    time.sleep(0.5)
                    return True
                else:
                    print("📭 Collection already empty")
                    return True
            except Exception as e:
                print(f"   ⚠️ Clear error: {e}")
                return False
        return False
    
    def close_connections(self):
        """Close any open connections to the database"""
        try:
            if self.vectorstore:
                self.vectorstore = None
                self.is_initialized = False
                print("🔌 Connections closed")
                time.sleep(1)
        except Exception as e:
            print(f"   ⚠️ Close connections error: {e}")
    
    def delete_persist_directory(self):
        """Delete the persistence directory completely"""
        self.close_connections()
        
        if os.path.exists(self.persist_dir):
            try:
                time.sleep(2)
                shutil.rmtree(self.persist_dir)
                print(f"🗑️ Deleted persist directory: {self.persist_dir}")
                return True
            except PermissionError:
                print(f"   ⚠️ Could not delete directory immediately (file locked)")
                try:
                    os.system(f'rmdir /s /q "{self.persist_dir}" 2>nul')
                    print(f"   ✅ Force deleted using system command")
                except:
                    pass
                return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.is_initialized or self.vectorstore is None:
            return {
                "initialized": False,
                "total_chunks": 0,
                "sources": []
            }
        
        try:
            results = self.vectorstore.get()
            return {
                "initialized": True,
                "total_chunks": len(results['ids']),
                "sources": self.get_all_sources(),
                "persist_dir": self.persist_dir,
                "collection": self.collection
            }
        except Exception:
            return {"initialized": False, "total_chunks": 0, "sources": []}