"""
Multi-Document Support
Handle multiple documents with separate collections
"""
from typing import Dict, List, Optional
import uuid
from datetime import datetime

class MultiDocumentManager:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.documents = {}  # doc_id -> document_info
        self.collections = {}  # doc_id -> collection_name
        
    def add_document(self, file_path: str, metadata: Optional[Dict] = None):
        """Add a document to the system"""
        doc_id = str(uuid.uuid4())[:8]
        collection_name = f"doc_{doc_id}"
        
        # Store document info
        self.documents[doc_id] = {
            'id': doc_id,
            'path': file_path,
            'name': file_path.split('/')[-1],
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'collection': collection_name
        }
        
        # Process and store in separate collection
        self.orchestrator.ingest_document(file_path, collection_name=collection_name)
        self.collections[doc_id] = collection_name
        
        return doc_id
    
    def ask_document(self, doc_id: str, question: str) -> Dict:
        """Ask a question to a specific document"""
        if doc_id not in self.collections:
            return {'error': 'Document not found'}
        
        # Query specific collection
        return self.orchestrator.ask(question, collection_name=self.collections[doc_id])
    
    def ask_all_documents(self, question: str) -> List[Dict]:
        """Ask a question to all documents"""
        results = []
        for doc_id, collection in self.collections.items():
            answer = self.orchestrator.ask(question, collection_name=collection)
            results.append({
                'document_id': doc_id,
                'document_name': self.documents[doc_id]['name'],
                'answer': answer
            })
        return results
    
    def compare_documents(self, question: str) -> Dict:
        """Compare answers from different documents"""
        results = self.ask_all_documents(question)
        
        # Find similarities and differences
        comparison = {
            'question': question,
            'results': results,
            'similarities': self._find_similarities(results),
            'differences': self._find_differences(results)
        }
        return comparison
    
    def _find_similarities(self, results: List[Dict]) -> List[str]:
        """Find similar answers across documents (simplified)"""
        # Simple implementation - can be enhanced
        answers = [r['answer']['answer'] for r in results]
        # Check if answers are similar (simplified)
        return [a for a in answers if len(set(a.split())) > 5]
    
    def _find_differences(self, results: List[Dict]) -> List[str]:
        """Find different perspectives across documents"""
        # Placeholder for difference detection
        return ["Documents provide different perspectives"]