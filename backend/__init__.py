# backend/__init__.py
"""
Backend package for Document Q&A System
"""

from backend.config import Config
from backend.orchestrator import DocumentQaOrchestrator
from backend.document_processor import DocumentProcessor
from backend.chroma_store import ChromaStoreManager
from backend.qa_chain import QaChain
from backend.cache_manager import CacheManager
from backend.feedback_system import FeedbackSystem

__all__ = [
    'Config',
    'DocumentQaOrchestrator',
    'DocumentProcessor',
    'ChromaStoreManager',
    'QaChain',
    'CacheManager',
    'FeedbackSystem'
]

__version__ = '2.0.0'