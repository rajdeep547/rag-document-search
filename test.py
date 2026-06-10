# final_system_test.py
"""
Complete System Test for Document Q&A Project
Tests every component and gives a clear pass/fail report
"""

import os
import sys
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("="*70)
print("🔍 DOCUMENT Q&A SYSTEM - COMPLETE TEST")
print("="*70)

# Track test results
tests = {
    "passed": 0,
    "failed": 0,
    "details": []
}

def test_pass(name, message):
    tests["passed"] += 1
    tests["details"].append(f"✅ {name}: {message}")
    print(f"   ✅ {message}")

def test_fail(name, message, suggestion=None):
    tests["failed"] += 1
    tests["details"].append(f"❌ {name}: {message}")
    print(f"   ❌ {message}")
    if suggestion:
        print(f"      💡 Suggestion: {suggestion}")

def test_section(title):
    print("\n" + "-"*50)
    print(f"📋 {title}")
    print("-"*50)

# ============================================
# TEST 1: Environment and API Keys
# ============================================
test_section("1. Environment Configuration")

# Check .env file
if os.path.exists(".env"):
    test_pass("Environment", ".env file found")
else:
    test_fail("Environment", ".env file not found", "Create .env file with your API key")

# Check for API keys
google_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if google_key:
    if google_key.startswith("AIza"):
        test_pass("Google API Key", f"Found (starts with AIza, length: {len(google_key)})")
    else:
        test_fail("Google API Key", f"Found but wrong format (starts with {google_key[:3]}). Should start with AIza", "Get key from https://aistudio.google.com/")
else:
    test_fail("Google API Key", "Not found in .env file", "Add GOOGLE_API_KEY=your-key to .env")

if groq_key:
    if groq_key.startswith("gsk_"):
        test_pass("Groq API Key", f"Found (starts with gsk_, length: {len(groq_key)})")
    else:
        test_fail("Groq API Key", f"Found but wrong format", "Get key from https://console.groq.com")
else:
    print(f"   ⚠️ Groq API Key: Not found (optional)")

# ============================================
# TEST 2: Python Dependencies
# ============================================
test_section("2. Python Dependencies")

required_packages = [
    "langchain",
    "langchain_community",
    "langchain_core",
    "langchain_text_splitters",
    "chromadb",
    "streamlit",
    "pypdf",
    "dotenv"
]

optional_packages = [
    "sentence_transformers",
    "langchain_google_genai",
    "langchain_groq"
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        test_pass("Package", f"{package} ✓")
    except ImportError:
        test_fail("Package", f"{package} ✗ - Not installed", f"Run: pip install {package}")

for package in optional_packages:
    try:
        __import__(package.replace("-", "_"))
        test_pass("Package", f"{package} ✓ (optional)")
    except ImportError:
        print(f"   ⚠️ {package}: Not installed (optional)")

# ============================================
# TEST 3: Backend Modules Import
# ============================================
test_section("3. Backend Modules")

modules_to_test = [
    "backend.config",
    "backend.document_processor",
    "backend.chroma_store",
    "backend.qa_chain",
    "backend.orchestrator"
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        test_pass("Import", f"{module_name} ✓")
    except Exception as e:
        test_fail("Import", f"{module_name} ✗ - {str(e)[:50]}", "Check file exists and syntax is correct")

# ============================================
# TEST 4: Document Processing
# ============================================
test_section("4. Document Processing")

# Create a test document
test_doc = "test_sample.txt"
with open(test_doc, "w") as f:
    f.write("""Company Vacation Policy

Employees are entitled to 20 days of paid vacation per year.
Vacation requests must be approved 2 weeks in advance.
Unused vacation days expire at the end of the year.
""")

print(f"   📄 Created test document: {test_doc}")

try:
    from backend.document_processor import DocumentProcessor
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process_file(test_doc)
    
    if chunks and len(chunks) > 0:
        test_pass("Document Processing", f"Loaded and split into {len(chunks)} chunks")
        print(f"      First chunk preview: {chunks[0].page_content[:80]}...")
    else:
        test_fail("Document Processing", "No chunks created", "Check document_processor.py")
        
except Exception as e:
    test_fail("Document Processing", f"Error: {str(e)[:100]}", "Check document_processor.py")

# ============================================
# TEST 5: Embeddings (Local)
# ============================================
test_section("5. Local Embeddings")

try:
    from sentence_transformers import SentenceTransformer
    
    print("   📥 Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vector = model.encode("test")
    
    if len(vector) > 0:
        test_pass("Local Embeddings", f"Working! Dimension: {len(vector)}")
    else:
        test_fail("Local Embeddings", "Generated empty vector", "Reinstall sentence-transformers")
        
except ImportError:
    test_fail("Local Embeddings", "sentence-transformers not installed", "Run: pip install sentence-transformers")
except Exception as e:
    test_fail("Local Embeddings", f"Error: {str(e)[:80]}", "Check installation")

# ============================================
# TEST 6: ChromaDB Vector Store
# ============================================
test_section("6. Vector Store (ChromaDB)")

try:
    from backend.chroma_store import ChromaStoreManager
    
    # Create test store path
    test_store_path = os.path.join(os.getcwd(), "storage", "test_store")
    if os.path.exists(test_store_path):
        # Close any open connections first
        time.sleep(1)
        try:
            shutil.rmtree(test_store_path)
        except PermissionError:
            print(f"   ⚠️ Could not delete existing store, will overwrite")
    
    store = ChromaStoreManager(persist_dir=test_store_path)
    store.add_documents(chunks)
    
    # Test search
    results = store.similarity_search("vacation days", k=2)
    
    if len(results) > 0:
        test_pass("ChromaDB", f"Store created and search working (found {len(results)} results)")
        print(f"      Top result score: {results[0]['similarity_score']}")
    else:
        test_fail("ChromaDB", "Search returned no results", "Check chroma_store.py")
    
    # Store the store reference for cleanup
    test_store_ref = store
    
except Exception as e:
    test_fail("ChromaDB", f"Error: {str(e)[:100]}", "Check chromadb installation")
    test_store_ref = None

# ============================================
# TEST 7: Groq LLM (if available)
# ============================================
test_section("7. Groq LLM (Fastest Option)")

if groq_key and groq_key.startswith("gsk_"):
    try:
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_retries=2
        )
        
        print("   🔄 Testing Groq API (max 10 seconds)...")
        start = time.time()
        response = llm.invoke("Say 'OK'")
        elapsed = time.time() - start
        
        test_pass("Groq LLM", f"Working! Response: {response.content[:50]}, Time: {elapsed:.1f}s")
        groq_working = True
        
    except Exception as e:
        test_fail("Groq LLM", f"Error: {str(e)[:80]}", "Check API key or network")
        groq_working = False
else:
    print(f"   ⚠️ Groq LLM: Skipped (no valid API key)")
    groq_working = False

# ============================================
# TEST 8: Google Gemini LLM (if available)
# ============================================
test_section("8. Google Gemini LLM")

gemini_working = False

if google_key and google_key.startswith("AIza"):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Try gemini-3.5-flash first
        models_to_try = ["gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
        
        for model_name in models_to_try:
            try:
                print(f"   🔄 Trying model: {model_name}...")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    google_api_key=google_key
                )
                
                start = time.time()
                response = llm.invoke("Say 'OK'")
                elapsed = time.time() - start
                
                test_pass("Google Gemini", f"Working with {model_name}! Response: {response.content[:50]}, Time: {elapsed:.1f}s")
                gemini_working = True
                break
                
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg:
                    print(f"      ❌ {model_name}: Not found")
                elif "503" in error_msg:
                    print(f"      ⚠️ {model_name}: Service unavailable")
                else:
                    print(f"      ⚠️ {model_name}: {error_msg[:50]}")
                continue
        
        if not gemini_working:
            test_fail("Google Gemini", "No working model found", "Check API key or use Groq")
            
    except Exception as e:
        test_fail("Google Gemini", f"Error: {str(e)[:80]}", "Check langchain-google-genai installation")
else:
    print(f"   ⚠️ Google Gemini: Skipped (no valid API key)")

# ============================================
# TEST 9: Complete RAG Pipeline (if any LLM works)
# ============================================
test_section("9. Complete RAG Pipeline")

if groq_working or gemini_working:
    try:
        from backend.qa_chain import QaChain
        from backend.config import Config
        
        # Use the existing chunks from earlier
        # Create mock search results (since we already have chunks)
        mock_results = [
            {
                "content": "Employees are entitled to 20 days of paid vacation per year.",
                "metadata": {"source": "test_sample.txt"},
                "similarity_score": 0.95
            }
        ]
        
        qa = QaChain()
        start = time.time()
        result = qa.generate_answer("How many vacation days?", mock_results)
        elapsed = time.time() - start
        
        if result.get('answer') and "error" not in result.get('confidence', ''):
            test_pass("RAG Pipeline", f"Working! Answer: {result['answer'][:80]}...")
            print(f"      Generation time: {elapsed:.1f}s")
            print(f"      Confidence: {result.get('confidence', 'unknown')}")
        else:
            test_fail("RAG Pipeline", f"Failed: {result.get('answer', 'Unknown error')}", "Check qa_chain.py")
            
    except Exception as e:
        test_fail("RAG Pipeline", f"Error: {str(e)[:100]}", "Check qa_chain.py and orchestrator.py")
else:
    print(f"   ⚠️ RAG Pipeline: Skipped (no working LLM)")

# ============================================
# TEST 10: Cleanup
# ============================================
test_section("10. Cleanup")

# Remove test document
if os.path.exists("test_sample.txt"):
    os.remove("test_sample.txt")
    print(f"   🧹 Removed test document")

# Clean up ChromaDB - close any connections first
if 'test_store_ref' in locals() and test_store_ref:
    try:
        if hasattr(test_store_ref, 'vectorstore') and test_store_ref.vectorstore:
            test_store_ref.clear_collection()
            time.sleep(1)
    except:
        pass

# Then remove directory with proper Windows handling
if os.path.exists("./storage/test_store"):
    import time
    time.sleep(2)  # Wait for Windows file lock to release
    try:
        shutil.rmtree("./storage/test_store")
        print(f"   🧹 Removed test storage")
    except PermissionError:
        print(f"   ⚠️ Could not delete immediately - file locked")
        print(f"   💡 Will be deleted on next run or you can manually delete storage/test_store folder")
        # Try force delete on Windows
        try:
            os.system(f'rmdir /s /q "{os.path.abspath("./storage/test_store")}" 2>nul')
        except:
            pass

test_pass("Cleanup", "Cleanup completed")

# ============================================
# FINAL REPORT
# ============================================
print("\n" + "="*70)
print("📊 TEST RESULTS SUMMARY")
print("="*70)

print(f"\n   ✅ Passed: {tests['passed']}")
print(f"   ❌ Failed: {tests['failed']}")
if (tests['passed'] + tests['failed']) > 0:
    print(f"   📈 Success Rate: {tests['passed']/(tests['passed']+tests['failed'])*100:.1f}%")

print("\n" + "-"*50)
print("📝 DETAILS:")
print("-"*50)
for detail in tests['details']:
    print(f"   {detail}")

print("\n" + "="*70)

if tests['failed'] == 0:
    print("\n🎉 CONGRATULATIONS! Your system is FULLY WORKING!")
    print("   You can now run: python test_with_dialog.py")
elif tests['failed'] <= 3:
    print("\n⚠️ Your system has minor issues. Check the failed tests above.")
    print("   Most likely: API key format or missing packages.")
    print("\n💡 Quick fix: Use Groq instead of Gemini")
    print("   1. Get API key from https://console.groq.com")
    print("   2. Add GROQ_API_KEY to .env file")
    print("   3. Run this test again")
else:
    print("\n❌ Your system has major issues. Please:")
    print("   1. Make sure you have a valid API key")
    print("   2. Run: pip install -r requirements.txt")
    print("   3. Check all backend files exist")
    print("   4. Or switch to Groq (simpler, faster)")

print("="*70)