# backend/orchestrator.py
from pathlib import Path
from backend.config import Config
from backend.document_processor import DocumentProcessor
from backend.chroma_store import ChromaStoreManager
from backend.qa_chain import QaChain

class DocumentQaOrchestrator:
    def __init__(self, config=None):
        self.config = config or Config()
        
        print("\n" + "="*60)
        print("🚀 Starting Document QA System")
        print("   Powered by Gemini 3.5 Flash")
        print("   ✦ LLM: Gemini 3.5 Flash")
        print("   ✦ Embeddings: gemini-embedding-001")
        print("="*60)
        
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
        
        self.ingested = set(self.store.get_all_sources())
        print(f"\n✅ System Ready!")
        print(f"   📚 Documents ingested: {len(self.ingested)}")
        print(f"   🔮 LLM: {self.config.LLM_MODEL}")
        print(f"   🔢 Embeddings: {self.config.EMBEDDING_MODEL}")
        print("="*60 + "\n")
    
    def ingest_document(self, file_path: str):
        name = Path(file_path).name
        print(f"\n📥 Ingesting: {name}")
        
        if name in self.ingested:
            return {"success": True, "message": f"Already ingested: {name}", "already_ingested": True}
        
        try:
            chunks = self.processor.process_file(file_path)
            if not chunks:
                return {"success": False, "message": "No content extracted from document"}
            
            self.store.add_documents(chunks)
            self.ingested.add(name)
            
            return {
                "success": True,
                "message": f"✅ Successfully ingested: {name}",
                "chunks_created": len(chunks),
                "filename": name
            }
        except Exception as e:
            return {"success": False, "message": f"Failed: {str(e)}", "error": str(e)}
    
    def ask(self, question: str):
        print(f"\n❓ Question: {question}")
        
        if not self.store.is_initialized:
            return {
                "answer": "No documents ingested yet. Please upload a document first.",
                "sources": [],
                "confidence": "none",
                "error": "no_documents"
            }
        
        docs = self.store.similarity_search(question, k=self.config.TOP_K_RESULTS)
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant information in the document to answer your question.",
                "sources": [],
                "confidence": "none"
            }
        
        return self.qa.generate_answer(question, docs)
    
    def get_status(self):
        return {
            "initialized": True,
            "total_files": len(self.ingested),
            "files": list(self.ingested),
            "store_initialized": self.store.is_initialized,
            "llm_model": self.config.LLM_MODEL,
            "embedding_model": self.config.EMBEDDING_MODEL,
            "chunk_size": self.config.CHUNK_SIZE,
            "top_k": self.config.TOP_K_RESULTS
        }
    
    def clear_all(self):
        self.store.clear_collection()
        self.ingested.clear()
        print("🗑️ All documents cleared from the system")