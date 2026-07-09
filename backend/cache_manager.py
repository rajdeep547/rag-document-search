# backend/cache_manager.py
"""
Redis-based caching for frequently asked questions
"""
import hashlib
import json
from typing import Optional, Dict, Any
import os
import time


class CacheManager:
    """
    Cache manager for storing and retrieving answers
    Supports Redis or in-memory fallback
    """
    
    def __init__(self):
        """Initialize cache manager"""
        self.enabled = False
        self.redis_client = None
        self.memory_cache = {}  # Fallback in-memory cache
        self.cache_ttl = 3600  # 1 hour default
        
        # Try to connect to Redis
        self._init_redis()
        
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            import redis
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            self.cache_ttl = int(os.getenv('CACHE_TTL', 3600))
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            print("✅ Redis cache connected")
            
        except ImportError:
            print("⚠️ Redis not installed, using in-memory cache")
            self.enabled = False
            
        except Exception as e:
            print(f"⚠️ Redis not available, using in-memory cache: {e}")
            self.enabled = False
    
    def _get_cache_key(self, question: str, document_hash: str = "") -> str:
        """
        Generate unique cache key
        
        Args:
            question: User question
            document_hash: Document identifier
            
        Returns:
            Cache key string
        """
        text = f"{question}_{document_hash}".lower().strip()
        return f"qa_cache:{hashlib.md5(text.encode()).hexdigest()}"
    
    def get(self, question: str, document_hash: str = "") -> Optional[Dict]:
        """
        Get cached answer if available
        
        Args:
            question: User question
            document_hash: Document identifier
            
        Returns:
            Cached answer or None
        """
        key = self._get_cache_key(question, document_hash)
        
        if self.enabled and self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    result = json.loads(cached)
                    result['cached_at'] = time.time()
                    return result
            except Exception:
                pass
        
        # Fallback to memory cache
        if key in self.memory_cache:
            result = self.memory_cache[key]
            # Check if expired
            if time.time() - result.get('cached_at', 0) < self.cache_ttl:
                result['cached_at'] = time.time()
                return result
            else:
                del self.memory_cache[key]
        
        return None
    
    def set(self, question: str, answer: Dict, document_hash: str = ""):
        """
        Cache the answer
        
        Args:
            question: User question
            answer: Answer dictionary
            document_hash: Document identifier
        """
        key = self._get_cache_key(question, document_hash)
        
        # Add cache metadata
        answer['cached_at'] = time.time()
        
        if self.enabled and self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    self.cache_ttl,
                    json.dumps(answer)
                )
                return
            except Exception:
                pass
        
        # Fallback to memory cache
        self.memory_cache[key] = answer
    
    def clear_cache(self, pattern: str = "qa_cache:*"):
        """
        Clear cache entries
        
        Args:
            pattern: Redis key pattern to match
        """
        if self.enabled and self.redis_client:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    print(f"🗑️ Cleared {len(keys)} cache entries from Redis")
            except Exception as e:
                print(f"⚠️ Failed to clear Redis cache: {e}")
        
        # Clear memory cache
        self.memory_cache.clear()
        print("🗑️ Cleared in-memory cache")
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            'enabled': self.enabled,
            'ttl': self.cache_ttl,
            'memory_cache_size': len(self.memory_cache)
        }
        
        if self.enabled and self.redis_client:
            try:
                keys = self.redis_client.keys("qa_cache:*")
                stats['redis_cache_size'] = len(keys)
            except Exception:
                stats['redis_cache_size'] = 'unknown'
        
        return stats
    
    def get_cached_count(self) -> int:
        """Get number of cached items"""
        if self.enabled and self.redis_client:
            try:
                return len(self.redis_client.keys("qa_cache:*"))
            except Exception:
                pass
        return len(self.memory_cache)
    
    def remove_document_cache(self, document_hash: str):
        """Remove all cached answers for a document"""
        pattern = f"qa_cache:*{document_hash}*"
        self.clear_cache(pattern)


class SimpleCache(CacheManager):
    """
    Simple in-memory cache without Redis
    """
    
    def __init__(self, ttl: int = 3600):
        self.enabled = True
        self.redis_client = None
        self.memory_cache = {}
        self.cache_ttl = ttl
        print("✅ Using in-memory cache")
    
    def _init_redis(self):
        # Override to disable Redis
        pass