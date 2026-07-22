# main.py - Enhanced Document Q&A System with Multi-Document, Hybrid Search, and Analytics
import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Optional
import json

# Load environment variables
load_dotenv()

# Check for Groq API key
if not os.getenv("GROQ_API_KEY"):
    print("="*70)
    print("❌ ERROR: No Groq API key found in .env file")
    print("="*70)
    print("\nPlease add to your .env file:")
    print("GROQ_API_KEY=your-groq-api-key-here")
    print("\nGet your free API key from: https://console.groq.com/keys")
    print("="*70)
    sys.exit(1)

print("✅ Groq API key found")

# Try to import tkinter for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("⚠️ tkinter not available. Using command line input.")

# Import backend modules
from backend.document_processor import DocumentProcessor
from backend.chroma_store import ChromaStoreManager
from backend.qa_chain import QaChain
from backend.config import Config
from backend.hybrid_retriever import HybridRetriever
from backend.multi_document import MultiDocumentManager
from backend.cache_manager import CacheManager
from backend.feedback_system import FeedbackSystem

# ============================================================================
# ENHANCED DOCUMENT MANAGER CLASS
# ============================================================================

class EnhancedDocumentQASystem:
    """Enhanced Document Q&A System with all new features"""
    
    def __init__(self):
        """Initialize the enhanced system"""
        print("\n🚀 Initializing Enhanced Document Q&A System...")
        
        # Core components
        self.processor = DocumentProcessor()
        self.qa_chain = QaChain()
        self.config = Config()
        
        # Enhanced components
        self.cache = CacheManager()
        self.feedback = FeedbackSystem()
        
        # Document management
        self.documents = {}  # doc_id -> document info
        self.stores = {}     # doc_id -> ChromaStoreManager
        self.active_doc_id = None
        
        # Session tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'total_questions': 0,
            'total_documents': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Multi-document support
        self.multi_doc = MultiDocumentManager(self)
        
        print("✅ System initialized successfully!")
        print(f"   📊 LLM: {self.config.LLM_MODEL}")
        print(f"   🔢 Embeddings: {self.config.EMBEDDING_MODEL}")
        print(f"   ⚡ Cache: {'Enabled' if self.cache.enabled else 'Disabled'}")
        print("")
    
    def select_file(self) -> Optional[str]:
        """Open file selection dialog"""
        if TKINTER_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            file_path = filedialog.askopenfilename(
                title="Select a Document",
                filetypes=[
                    ("All supported", "*.pdf *.txt *.md *.docx"),
                    ("PDF files", "*.pdf"),
                    ("Text files", "*.txt"),
                    ("Markdown files", "*.md"),
                    ("Word documents", "*.docx"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            return file_path if file_path else None
        else:
            return input("Enter file path: ").strip().strip('"')
    
    def ingest_document(self, file_path: str) -> Dict:
        """
        Ingest a document into the system
        Returns: Document info dictionary
        """
        if not os.path.exists(file_path):
            return {'success': False, 'error': 'File not found'}
        
        filename = Path(file_path).name
        
        # Check if document already exists
        for doc_id, info in self.documents.items():
            if info['name'] == filename:
                return {
                    'success': False, 
                    'error': f'Document "{filename}" already exists',
                    'doc_id': doc_id
                }
        
        print(f"\n📄 Processing: {filename}")
        
        try:
            # Step 1: Process document
            print("   📖 Loading and chunking...")
            chunks = self.processor.process_file(file_path)
            
            if not chunks:
                return {'success': False, 'error': 'No content extracted'}
            
            print(f"   ✅ Created {len(chunks)} chunks")
            
            # Step 2: Create vector store
            print("   🗄️ Creating vector store...")
            doc_id = f"doc_{int(time.time())}_{len(self.documents)}"
            persist_dir = f"./storage/db_{doc_id}"
            
            store = ChromaStoreManager(persist_dir=persist_dir)
            store.add_documents(chunks)
            
            # Step 3: Store document info
            self.documents[doc_id] = {
                'id': doc_id,
                'name': filename,
                'path': file_path,
                'chunks': len(chunks),
                'store': store,
                'persist_dir': persist_dir,
                'ingested_at': datetime.now().isoformat(),
                'size': os.path.getsize(file_path)
            }
            
            self.stores[doc_id] = store
            self.active_doc_id = doc_id
            self.session_stats['total_documents'] += 1
            
            print(f"   ✅ Document ingested successfully!")
            print(f"   📊 Document ID: {doc_id}")
            
            return {
                'success': True,
                'doc_id': doc_id,
                'filename': filename,
                'chunks': len(chunks),
                'message': f'Successfully ingested "{filename}"'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def ask_question(self, question: str, doc_id: Optional[str] = None) -> Dict:
        """
        Ask a question to the document
        """
        if not question.strip():
            return {'error': 'Please enter a question'}
        
        # Use active document if no doc_id specified
        if doc_id is None:
            doc_id = self.active_doc_id
        
        if doc_id not in self.documents:
            return {'error': 'No document loaded. Please ingest a document first.'}
        
        # Check cache first
        doc_name = self.documents[doc_id]['name']
        cache_key = f"{question}_{doc_id}"
        
        cached_answer = self.cache.get(cache_key, doc_id)
        if cached_answer:
            self.session_stats['cache_hits'] += 1
            cached_answer['from_cache'] = True
            cached_answer['cache_hit'] = True
            return cached_answer
        
        self.session_stats['cache_misses'] += 1
        
        # Search for relevant chunks
        store = self.documents[doc_id]['store']
        
        try:
            # Use hybrid search
            docs = store.similarity_search(question, k=self.config.TOP_K_RESULTS)
            
            if not docs:
                return {
                    'question': question,
                    'answer': 'No relevant information found in the document.',
                    'sources': [],
                    'confidence': 'low',
                    'from_cache': False
                }
            
            # Generate answer
            result = self.qa_chain.generate_answer(question, docs)
            
            # Add document info
            result['doc_id'] = doc_id
            result['doc_name'] = self.documents[doc_id]['name']
            result['from_cache'] = False
            
            # Cache the answer
            self.cache.set(cache_key, result, doc_id)
            
            # Update stats
            self.session_stats['total_questions'] += 1
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def summarize_document(self, doc_id: Optional[str] = None, length: str = "medium") -> Dict:
        """Summarize a document by ID (defaults to the active document)"""
        if doc_id is None:
            doc_id = self.active_doc_id

        if doc_id not in self.documents:
            return {'error': 'No document loaded. Please ingest a document first.'}

        store = self.documents[doc_id]['store']
        chunks = store.get_all_chunks()

        if not chunks:
            return {'error': 'No content found for this document.'}

        result = self.qa_chain.generate_summary(chunks, doc_name=self.documents[doc_id]['name'], length=length)
        result['doc_id'] = doc_id
        result['doc_name'] = self.documents[doc_id]['name']
        return result

    def ask_all_documents(self, question: str) -> List[Dict]:
        """Ask question to all documents"""
        results = []
        for doc_id in self.documents:
            result = self.ask_question(question, doc_id)
            results.append({
                'doc_id': doc_id,
                'doc_name': self.documents[doc_id]['name'],
                'answer': result
            })
        return results
    
    def compare_documents(self, question: str) -> Dict:
        """Compare answers across documents"""
        results = self.ask_all_documents(question)
        
        comparison = {
            'question': question,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return comparison
    
    def show_status(self):
        """Show system status"""
        print("\n" + "="*60)
        print("📊 SYSTEM STATUS")
        print("="*60)
        print(f"\n📚 Active Documents: {len(self.documents)}")
        for doc_id, info in self.documents.items():
            print(f"   📄 {info['name']} (ID: {doc_id})")
            print(f"      - Chunks: {info['chunks']}")
            print(f"      - Size: {info['size'] / 1024:.1f} KB")
        
        print(f"\n💬 Total Questions: {self.session_stats['total_questions']}")
        print(f"⚡ Cache Hits: {self.session_stats['cache_hits']}")
        print(f"📊 Cache Misses: {self.session_stats['cache_misses']}")
        print(f"⏱️  Cache Ratio: {self._get_cache_ratio():.1%}")
        
        print(f"\n🔧 Active Document: {self.documents.get(self.active_doc_id, {}).get('name', 'None')}")
        print(f"🤖 LLM Model: {self.config.LLM_MODEL}")
        print(f"🔢 Embedding Model: {self.config.EMBEDDING_MODEL}")
        
        # Feedback stats
        feedback_stats = self.feedback.get_stats()
        if feedback_stats['total_feedback'] > 0:
            print(f"\n⭐ Average Rating: {feedback_stats['average_rating']:.1f}/5")
            print(f"💬 Total Feedback: {feedback_stats['total_feedback']}")
        
        print("="*60)
    
    def _get_cache_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total = self.session_stats['cache_hits'] + self.session_stats['cache_misses']
        if total == 0:
            return 0.0
        return self.session_stats['cache_hits'] / total
    
    def show_feedback_stats(self):
        """Show feedback statistics"""
        print("\n" + "="*60)
        print("📊 FEEDBACK STATISTICS")
        print("="*60)
        
        stats = self.feedback.get_stats()
        print(f"\n📝 Total Feedback: {stats['total_feedback']}")
        print(f"⭐ Average Rating: {stats['average_rating']:.1f}/5")
        
        if stats['rating_distribution']:
            print("\n📊 Rating Distribution:")
            for rating, count in sorted(stats['rating_distribution'].items()):
                bar = "█" * (count * 2)
                print(f"   {rating}★: {bar} ({count})")
        
        if stats['top_questions']:
            print("\n🔥 Top Questions:")
            for q, count in list(stats['top_questions'].items())[:5]:
                print(f"   📌 {q[:50]}... ({count} times)")
        
        print("\n💡 Improvement Suggestions:")
        suggestions = self.feedback.get_improvement_suggestions()
        for suggestion in suggestions:
            print(f"   • {suggestion}")
        
        print("="*60)
    
    def export_session(self, filename: str = "session_export.json"):
        """Export session data"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'documents': {
                doc_id: {
                    'name': info['name'],
                    'chunks': info['chunks'],
                    'size': info['size'],
                    'ingested_at': info['ingested_at']
                }
                for doc_id, info in self.documents.items()
            },
            'stats': self.session_stats,
            'feedback': self.feedback.feedback_data
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Session exported to {filename}")
    
    def clear_cache(self):
        """Clear all cached answers"""
        self.cache.clear_cache()
        print("✅ Cache cleared")
    
    def remove_document(self, doc_id: str):
        """Remove a document from the system"""
        if doc_id not in self.documents:
            print(f"❌ Document {doc_id} not found")
            return
        
        # Remove from storage
        persist_dir = self.documents[doc_id]['persist_dir']
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        
        # Remove from tracking
        del self.documents[doc_id]
        del self.stores[doc_id]
        
        if self.active_doc_id == doc_id:
            self.active_doc_id = list(self.documents.keys())[0] if self.documents else None
        
        print(f"✅ Document removed successfully")


# ============================================================================
# MAIN INTERACTIVE CLI
# ============================================================================

def main():
    """Main interactive CLI"""
    print("="*70)
    print("📚 ENHANCED DOCUMENT Q&A SYSTEM")
    print("   Built with: Groq (Llama 3.3 70B) + ChromaDB + RAG")
    print("   Features: Multi-Doc, Hybrid Search, Caching, Analytics")
    print("="*70)
    
    # Initialize system
    system = EnhancedDocumentQASystem()
    
    # Interactive menu
    while True:
        print("\n" + "-"*70)
        print("📋 MAIN MENU")
        print("-"*70)
        print("1. 📄 Ingest Document")
        print("2. 💬 Ask Question")
        print("3. 📚 Ask All Documents")
        print("4. 🔄 Compare Documents")
        print("5. 📊 Show Status")
        print("6. ⭐ Feedback Statistics")
        print("7. 📥 Export Session")
        print("8. 🗑️ Clear Cache")
        print("9. 🗑️ Remove Document")
        print("10. 🧹 Clear Screen")
        print("11. 📝 Summarize Document")
        print("12. 🚪 Exit")
        print("-"*70)

        choice = input("Select option (1-12): ").strip()
        
        # ====================================================================
        # 1. Ingest Document
        # ====================================================================
        if choice == "1":
            print("\n📁 Please select a document...")
            file_path = system.select_file()
            
            if not file_path:
                print("❌ No file selected")
                continue
            
            result = system.ingest_document(file_path)
            if result['success']:
                print(f"\n✅ {result['message']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        
        # ====================================================================
        # 2. Ask Question
        # ====================================================================
        elif choice == "2":
            if not system.documents:
                print("\n❌ No documents loaded. Please ingest a document first.")
                continue
            
            # Select document
            print("\n📚 Available Documents:")
            doc_list = list(system.documents.items())
            for i, (doc_id, info) in enumerate(doc_list, 1):
                active = " (active)" if doc_id == system.active_doc_id else ""
                print(f"   {i}. {info['name']}{active}")
            
            print(f"   {len(doc_list) + 1}. Ask all documents")
            
            doc_choice = input("\nSelect document (number): ").strip()
            
            if doc_choice.isdigit():
                idx = int(doc_choice) - 1
                if idx < len(doc_list):
                    doc_id = doc_list[idx][0]
                elif idx == len(doc_list):
                    # Ask all documents
                    question = input("\n❓ Your question: ").strip()
                    if question:
                        results = system.ask_all_documents(question)
                        print(f"\n📝 Results from {len(results)} documents:")
                        for result in results:
                            print(f"\n📄 {result['doc_name']}:")
                            if 'error' in result['answer']:
                                print(f"   ❌ {result['answer']['error']}")
                            else:
                                print(f"   🤖 {result['answer'].get('answer', 'No answer')}")
                                if result['answer'].get('confidence'):
                                    print(f"   📊 Confidence: {result['answer']['confidence']}")
                    continue
                else:
                    print("❌ Invalid selection")
                    continue
            else:
                print("❌ Invalid input")
                continue
            
            # Ask question to specific document
            question = input("\n❓ Your question: ").strip()
            if question:
                result = system.ask_question(question, doc_id)
                if 'error' in result:
                    print(f"\n❌ {result['error']}")
                else:
                    print(f"\n🤖 Answer: {result.get('answer', 'No answer')}")
                    if result.get('sources'):
                        print(f"📎 Sources: {', '.join(result['sources'])}")
                    if result.get('confidence'):
                        print(f"📊 Confidence: {result['confidence']}")
                    if result.get('generation_time'):
                        print(f"⏱️ Generation time: {result['generation_time']}s")
                    if result.get('from_cache'):
                        print("⚡ (Answer from cache)")
                    
                    # Collect feedback
                    feedback = input("\nWas this answer helpful? (y/n): ").strip().lower()
                    if feedback in ['y', 'yes']:
                        system.feedback.add_feedback(
                            question,
                            result.get('answer', ''),
                            result.get('doc_name', 'Unknown'),
                            5
                        )
                        print("✅ Thanks for your feedback!")
                    elif feedback in ['n', 'no']:
                        system.feedback.add_feedback(
                            question,
                            result.get('answer', ''),
                            result.get('doc_name', 'Unknown'),
                            1
                        )
                        print("✅ We'll work on improving!")
        
        # ====================================================================
        # 3. Ask All Documents
        # ====================================================================
        elif choice == "3":
            if not system.documents:
                print("\n❌ No documents loaded.")
                continue
            
            question = input("\n❓ Your question for all documents: ").strip()
            if question:
                results = system.ask_all_documents(question)
                print(f"\n📝 Results from {len(results)} documents:")
                for result in results:
                    print(f"\n📄 {result['doc_name']}:")
                    if 'error' in result['answer']:
                        print(f"   ❌ {result['answer']['error']}")
                    else:
                        print(f"   🤖 {result['answer'].get('answer', 'No answer')}")
                        if result['answer'].get('confidence'):
                            print(f"   📊 Confidence: {result['answer']['confidence']}")
        
        # ====================================================================
        # 4. Compare Documents
        # ====================================================================
        elif choice == "4":
            if len(system.documents) < 2:
                print("\n❌ Need at least 2 documents for comparison.")
                continue
            
            question = input("\n❓ Question to compare: ").strip()
            if question:
                comparison = system.compare_documents(question)
                print(f"\n📊 Comparison Results:")
                print(f"Question: {comparison['question']}\n")
                
                for result in comparison['results']:
                    print(f"📄 {result['doc_name']}:")
                    if 'error' in result['answer']:
                        print(f"   ❌ {result['answer']['error']}")
                    else:
                        print(f"   🤖 {result['answer'].get('answer', 'No answer')}")
                        print(f"   📊 Confidence: {result['answer'].get('confidence', 'N/A')}")
                    print()
        
        # ====================================================================
        # 5. Show Status
        # ====================================================================
        elif choice == "5":
            system.show_status()
        
        # ====================================================================
        # 6. Feedback Statistics
        # ====================================================================
        elif choice == "6":
            system.show_feedback_stats()
        
        # ====================================================================
        # 7. Export Session
        # ====================================================================
        elif choice == "7":
            filename = input("Export filename (default: session_export.json): ").strip()
            if not filename:
                filename = "session_export.json"
            system.export_session(filename)
        
        # ====================================================================
        # 8. Clear Cache
        # ====================================================================
        elif choice == "8":
            system.clear_cache()
        
        # ====================================================================
        # 9. Remove Document
        # ====================================================================
        elif choice == "9":
            if not system.documents:
                print("\n❌ No documents to remove.")
                continue
            
            print("\n📚 Available Documents:")
            doc_list = list(system.documents.items())
            for i, (doc_id, info) in enumerate(doc_list, 1):
                print(f"   {i}. {info['name']}")
            
            doc_choice = input("Select document to remove (number): ").strip()
            if doc_choice.isdigit():
                idx = int(doc_choice) - 1
                if 0 <= idx < len(doc_list):
                    doc_id = doc_list[idx][0]
                    confirm = input(f"Remove {system.documents[doc_id]['name']}? (y/n): ").strip().lower()
                    if confirm == 'y':
                        system.remove_document(doc_id)
                else:
                    print("❌ Invalid selection")
        
        # ====================================================================
        # 10. Clear Screen
        # ====================================================================
        elif choice == "10":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("="*70)
            print("📚 ENHANCED DOCUMENT Q&A SYSTEM")
            print("="*70)
        
        # ====================================================================
        # 11. Summarize Document
        # ====================================================================
        elif choice == "11":
            if not system.documents:
                print("\n❌ No documents loaded. Please ingest a document first.")
                continue

            print("\n📚 Available Documents:")
            doc_list = list(system.documents.items())
            for i, (doc_id, info) in enumerate(doc_list, 1):
                active = " (active)" if doc_id == system.active_doc_id else ""
                print(f"   {i}. {info['name']}{active}")

            doc_choice = input("\nSelect document (number, or Enter for active): ").strip()
            if not doc_choice:
                doc_id = system.active_doc_id
            elif doc_choice.isdigit() and 0 <= int(doc_choice) - 1 < len(doc_list):
                doc_id = doc_list[int(doc_choice) - 1][0]
            else:
                print("❌ Invalid selection")
                continue

            length = input("Summary length (short/medium/long) [medium]: ").strip().lower() or "medium"
            if length not in ("short", "medium", "long"):
                length = "medium"

            print(f"\n📝 Summarizing {system.documents[doc_id]['name']}...")
            result = system.summarize_document(doc_id, length=length)
            if 'error' in result:
                print(f"\n❌ {result['error']}")
            else:
                print(f"\n📄 Summary of {result.get('doc_name', 'document')}:\n")
                print(result.get('summary', 'No summary generated'))
                print(f"\n⏱️ Generated in {result.get('generation_time', 0)}s ({result.get('chunks_used', 0)} chunks)")

        # ====================================================================
        # 12. Exit
        # ====================================================================
        elif choice == "12":
            print("\n👋 Goodbye!")
            print(f"📊 Session Summary:")
            print(f"   - Total Documents: {system.session_stats['total_documents']}")
            print(f"   - Total Questions: {system.session_stats['total_questions']}")
            print(f"   - Cache Hits: {system.session_stats['cache_hits']}")
            print(f"   - Cache Ratio: {system._get_cache_ratio():.1%}")
            break

        else:
            print("❌ Invalid choice. Please select 1-12.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please report this issue.")