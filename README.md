# 📚 RAG Document Search System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **🤖 Chat with your documents like magic!** Upload PDFs, Word docs, or text files and ask natural language questions. Get AI-powered answers with **source citations** in seconds.

**Stop reading documents—let them answer you.**

---

## 🎯 What Does This Project Do?

This is an **intelligent document question-answering system**. In plain English:

1. **You upload** any document (PDF, Word, Text, Markdown, CSV).
2. **You ask a question** in plain English (e.g., "What were the key findings in the Q3 report?").
3. **The AI finds the answer** by searching through your document's content.
4. **You get a clear answer** with a link to the exact source and a confidence score.

Under the hood, it uses **RAG (Retrieval-Augmented Generation)**, which combines a powerful search engine with Google's Gemini LLM to give you accurate, contextual answers.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-Format Support** | Upload PDF, DOCX, TXT, Markdown, and CSV files |
| 🔍 **Semantic Search** | Finds information based on meaning, not just keywords |
| 💬 **Natural Language Q&A** | Ask questions in plain English |
| 📎 **Source Citations** | See exactly which document and section the answer came from |
| 📊 **Confidence Scoring** | Know how confident the AI is (High/Medium/Low) |
| 🖥️ **User-Friendly Interface** | Drag & drop file upload with Streamlit |
| ⚡ **Fast Responses** | Powered by Google's Gemini 1.5 Flash |
| 📈 **Analytics Dashboard** | Track usage, performance, and feedback |
| ⭐ **Feedback System** | Rate answers and help the system improve |
| 🔄 **Multi-Document Support** | Query across multiple documents at once |
| 🔒 **Secure** | API keys are never exposed |

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────────┐
│ YOUR DOCUMENT Q&A SYSTEM │
├─────────────────────────────────────────────────────────────────┤
│ │
│ 1. 📄 Document Upload ──► 2. 🔧 Processing & Chunking │
│ (PDF, DOCX, TXT) (Text is split into segments) │
│ │
│ │ │
│ ▼ │
│ │
│ 3. 🔢 Embeddings Created ──► 4. 💾 Stored in Vector DB │
│ (Converted to vectors) (ChromaDB) │
│ │
│ │
│ 5. ❓ User Asks Question ──► 6. 🔍 Semantic Search │
│ (Natural Language) (Finds relevant chunks) │
│ │
│ │ │
│ ▼ │
│ │
│ 7. 🤖 Gemini LLM Generates ──► 8. 📝 Answer + Citations │
│ (Context-aware answer) (Displayed in the UI) │
│ │
└─────────────────────────────────────────────────────────────────┘

text

---

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **AI Model** | Google Gemini 1.5 Flash | Core LLM for generating answers |
| **LLM Framework** | LangChain | Orchestrates LLM and vector database |
| **Vector Database** | ChromaDB | Stores and retrieves document embeddings |
| **Web Interface** | Streamlit | Interactive, user-friendly UI |
| **Embeddings** | gemini-embedding-001 | Converts text to vectors |
| **Search** | Hybrid Search | Semantic + keyword (BM25) search |
| **Caching** | Redis (Optional) | Faster response times for repeated queries |
| **Language** | Python 3.9+ | Backend and core logic |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Google Gemini API key (free - get it from [Google AI Studio](https://aistudio.google.com/))

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/rajdeep547/rag-document-search.git
cd rag-document-search
