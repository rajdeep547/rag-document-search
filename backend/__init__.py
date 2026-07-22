# backend/__init__.py
"""
Backend package for Document Q&A System
"""
import sys

# Windows consoles default to a non-UTF-8 codepage, which can't print
# the emoji used in this package's status/log messages.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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