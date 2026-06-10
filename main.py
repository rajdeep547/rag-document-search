# test_with_dialog.py - Test with real document from file dialog (Gemini 3.5 Flash)
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Check for Google Gemini API key (not Groq)
if not os.getenv("GOOGLE_API_KEY"):
    print("="*60)
    print("❌ ERROR: No Google Gemini API key found in .env file")
    print("="*60)
    print("\nPlease add to your .env file:")
    print("GOOGLE_API_KEY=your-gemini-api-key-here")
    print("\nGet your free API key from: https://aistudio.google.com/")
    print("="*60)
    sys.exit(1)

print("✅ Google Gemini API key found")

# Try to import tkinter for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("⚠️ tkinter not available. Using command line input.")

def select_file():
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

# Import backend modules (Gemini versions)
from backend.document_processor import DocumentProcessor
from backend.chroma_store import ChromaStoreManager
from backend.qa_chain import QaChain
from backend.config import Config

print("="*60)
print("📚 Document Q&A System Test (Gemini 3.5 Flash)")
print("   LLM: Gemini 3.5 Flash | Embeddings: gemini-embedding-001")
print("="*60)

# Select file
print("\n📁 Please select a document...")
file_path = select_file()

if not file_path:
    print("❌ No file selected. Exiting...")
    sys.exit(1)

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    sys.exit(1)

filename = Path(file_path).name
print(f"\n✅ Selected: {filename}")

# Step 1: Process document
print("\n" + "-"*40)
print("📄 Step 1: Processing Document")
print("-"*40)

try:
    processor = DocumentProcessor()
    print("   Loading and chunking...")
    chunks = processor.process_file(file_path)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    if not chunks:
        print("   ❌ No content extracted from document")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Step 2: Create vector store (using Gemini embeddings)
print("\n" + "-"*40)
print("🗄️ Step 2: Creating Vector Store (Gemini Embeddings)")
print("-"*40)

try:
    # Use a temporary test database
    import shutil
    test_db_path = "./storage/test_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
    
    store = ChromaStoreManager(persist_dir=test_db_path)
    store.add_documents(chunks)
    print("   ✅ Vector store created")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Step 3: Test search
print("\n" + "-"*40)
print("🔍 Step 3: Testing Search")
print("-"*40)

test_question = "What is this document about?"
print(f"   Test question: '{test_question}'")

try:
    results = store.similarity_search(test_question, k=3)
    print(f"   ✅ Found {len(results)} relevant chunks")
    for i, r in enumerate(results):
        print(f"      [{i+1}] Score: {r['similarity_score']:.3f}")
        print(f"          Preview: {r['content'][:100]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Step 4: Test QA chain (using Gemini 3.5 Flash LLM)
print("\n" + "-"*40)
print("💭 Step 4: Testing Q&A Generation (Gemini 3.5 Flash LLM)")
print("-"*40)

try:
    qa = QaChain()
    print("   Initialized QA chain with Gemini 3.5 Flash")
    
    # Use the search results
    answer = qa.generate_answer(test_question, results)
    print(f"\n📝 Answer: {answer['answer']}")
    print(f"   Sources: {answer['sources']}")
    print(f"   Confidence: {answer['confidence']}")
    if answer.get('generation_time'):
        print(f"   ⏱️ Generation time: {answer['generation_time']}s")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Step 5: Interactive Q&A
print("\n" + "-"*40)
print("💬 Step 5: Interactive Q&A (Ask questions about your document)")
print("-"*40)
print("Type your questions (or 'quit' to exit)\n")

question_count = 0
while True:
    question = input("❓ You: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        break
    
    if question.lower() == 'status':
        print(f"\n📊 Session Status:")
        print(f"   - Document: {filename}")
        print(f"   - Questions asked: {question_count}")
        print(f"   - LLM: Gemini 3.5 Flash")
        print(f"   - Embeddings: gemini-embedding-001")
        continue
    
    if question.lower() == 'clear':
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*60)
        print("📚 Document Q&A System (Gemini 3.5 Flash)")
        print("="*60)
        print(f"\n📄 Active Document: {filename}")
        print("\n💬 Ask your questions (type 'quit' to exit)\n")
        continue
    
    if not question:
        continue
    
    print("   🤔 Thinking with Gemini 3.5 Flash...", end="", flush=True)
    
    try:
        # Search for relevant chunks
        docs = store.similarity_search(question, k=5)
        
        if not docs:
            print("\r" + " " * 40 + "\r", end="")
            print("❌ No relevant information found in the document.")
            continue
        
        # Generate answer
        result = qa.generate_answer(question, docs)
        print("\r" + " " * 40 + "\r", end="")
        
        question_count += 1
        print(f"\n🤖 Answer: {result['answer']}")
        if result['sources']:
            print(f"📎 Sources: {', '.join(result['sources'])}")
        print(f"📊 Confidence: {result['confidence']}")
        if result.get('generation_time'):
            print(f"⏱️ Generation time: {result['generation_time']}s")
        print()
        
    except Exception as e:
        print("\r" + " " * 40 + "\r", end="")
        print(f"❌ Error: {e}")

# Clean up
print("\n" + "-"*40)
print("🧹 Cleaning up test database...")
print("-"*40)

import shutil
if os.path.exists("./storage/test_db"):
    shutil.rmtree("./storage/test_db")
    print("✅ Test database cleaned up")

print("\n" + "="*60)
print(f"📊 Session Summary:")
print(f"   - Document: {filename}")
print(f"   - Questions asked: {question_count}")
print(f"   - LLM: Gemini 3.5 Flash")
print(f"   - Embeddings: gemini-embedding-001")
print("="*60)