# 📚 RAG Document Search System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Chat with your documents** - Upload PDFs, Word docs, text files, and ask natural language questions with AI-powered answers and source citations.

An intelligent document question-answering system powered by **Google Gemini 1.5 Flash** with RAG (Retrieval-Augmented Generation) architecture, ChromaDB vector storage, and semantic search.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-Format Support** | Upload PDF, DOCX, TXT, Markdown, and CSV files |
| 🔍 **Semantic Search** | Uses embeddings to find relevant information |
| 💬 **Natural Language Q&A** | Ask questions in plain English |
| 📎 **Source Citations** | See exactly which document provided the answer |
| 📊 **Confidence Scoring** | Know how confident the AI is about each answer |
| 🖥️ **User-Friendly Interface** | Drag & drop file upload with Streamlit |
| ⚡ **Fast Responses** | Powered by Google's Gemini 1.5 Flash |
| 📈 **Analytics Dashboard** | Track usage, performance, and feedback |
| ⭐ **Feedback System** | Rate answers and improve the system |
| 🔄 **Multi-Document Support** | Query across multiple documents |
| 🔒 **Secure** | API keys never exposed |

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│ DOCUMENT Q&A SYSTEM │
├─────────────────────────────────────────────────────────────┤
│ │
│ 📄 Document Upload → 🔧 Processing → 🔢 Embeddings │
│ ↓ │
│ 💾 ChromaDB │
│ ↓ │
│ ❓ User Question → 🔍 Search → 🤖 LLM → 📝 Answer │
│ │
└─────────────────────────────────────────────────────────────┘

text

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Google Gemini API key (free)

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/rajdeep547/rag-document-search.git
cd rag-document-search
