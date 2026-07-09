# backend/qa_chain.py
"""
QA Chain using new google-genai SDK
"""
from typing import List, Dict, Optional
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class QaChain:
    """
    QA Chain using Gemini with the new google-genai SDK
    """
    
    def __init__(self, model="gemini-1.5-flash", temperature=0):
        """
        Initialize QA Chain
        
        Args:
            model: Gemini model name (default: gemini-1.5-flash)
            temperature: Temperature for generation (0 = factual)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        self.api_key = api_key
        self.temperature = temperature
        
        # Primary model and fallbacks - Using correct model names
        self.models = [
            "gemini-1.5-flash",  # Primary - Fast and efficient
            "gemini-1.5-pro",    # Fallback 1 - More capable
            "gemini-pro"         # Fallback 2 - Legacy stable
        ]
        self.current_model_index = 0
        self.current_model = self.models[0]
        
        # Initialize the new client
        self.client = genai.Client(api_key=api_key)
        
        self.request_count = 0
        self.last_request_time = 0
        print(f"✅ QaChain ready with {self.current_model}")
        print(f"   🔄 Fallback models: {', '.join(self.models[1:])}")
    
    def _try_fallback_models(self) -> bool:
        """Try to switch to a fallback model"""
        for i in range(self.current_model_index + 1, len(self.models)):
            model_name = self.models[i]
            print(f"   🔄 Trying fallback: {model_name}")
            self.current_model = model_name
            self.current_model_index = i
            return True
        return False
    
    def generate_answer(self, question: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer from retrieved chunks with retry logic
        
        Args:
            question: User question
            chunks: List of retrieved chunks with content and metadata
            
        Returns:
            Dictionary with answer, sources, confidence
        """
        start_time = time.time()
        
        if not chunks:
            return {
                "answer": "No relevant information found in the document.",
                "sources": [],
                "confidence": "low",
                "generation_time": 0
            }
        
        max_retries = len(self.models)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Prepare context from chunks
                context = self._prepare_context(chunks)
                
                # Create prompt
                prompt = f"""You are a helpful document assistant. Answer based ONLY on the context.

CONTEXT:
{context}

INSTRUCTIONS:
1. Answer ONLY using the context above
2. If the context doesn't have the answer, say: "I cannot find this information in the document."
3. Be concise and direct (2-3 sentences max)
4. If you find the answer, provide it clearly

Question: {question}

ANSWER:"""
                
                # Generate answer using the new SDK
                response = self.client.models.generate_content(
                    model=self.current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=2048,
                    )
                )
                
                answer = response.text.strip()
                
                # Extract sources
                sources = self._extract_sources(chunks)
                
                # Calculate confidence
                confidence = self._calculate_confidence(chunks, answer)
                
                generation_time = time.time() - start_time
                
                return {
                    "answer": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "generation_time": round(generation_time, 2),
                    "chunks_used": len(chunks),
                    "model_used": self.current_model
                }
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ Error with {self.current_model}: {error_msg[:100]}")
                
                # Check if it's a model not found or unavailable error
                if "404" in error_msg or "NOT_FOUND" in error_msg:
                    if self._try_fallback_models():
                        retry_count += 1
                        continue
                    else:
                        return {
                            "answer": "The AI model is not available. Please check your API key and model permissions.",
                            "sources": [],
                            "confidence": "low",
                            "generation_time": 0,
                            "error": "model_not_found"
                        }
                elif "503" in error_msg or "UNAVAILABLE" in error_msg or "high demand" in error_msg:
                    if self._try_fallback_models():
                        retry_count += 1
                        continue
                    else:
                        return {
                            "answer": "The AI service is currently experiencing high demand. Please try again in a few moments.",
                            "sources": [],
                            "confidence": "low",
                            "generation_time": 0,
                            "error": "service_unavailable"
                        }
                else:
                    return {
                        "answer": f"Error: {error_msg[:100]}",
                        "sources": [],
                        "confidence": "low",
                        "generation_time": 0,
                        "error": error_msg
                    }
        
        # All retries failed
        return {
            "answer": "Unable to generate answer. Please try again later.",
            "sources": [],
            "confidence": "low",
            "generation_time": 0,
            "error": "all_models_failed"
        }
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('content', '')
            if content:
                context_parts.append(f"[{i}] {content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        """Extract source names from chunks"""
        sources = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            if source and source not in sources:
                sources.append(source)
        return sources
    
    def _calculate_confidence(self, chunks: List[Dict], answer: str) -> str:
        """Calculate confidence based on chunks and answer"""
        if not chunks:
            return "low"
        
        # Check average similarity score
        scores = []
        for chunk in chunks:
            score = chunk.get('similarity_score', 0)
            if score > 0:
                scores.append(score)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.7:
                return "high"
            elif avg_score > 0.4:
                return "medium"
            else:
                return "low"
        
        # Check if answer is relevant
        if answer and len(answer) > 20:
            return "medium"
        
        return "low"
    
    def answer_question(self, question: str, chunks: List[Dict]) -> Dict:
        """Alias for generate_answer"""
        return self.generate_answer(question, chunks)
    
    def get_response(self, question: str, chunks: List[Dict]) -> Dict:
        """Alias for generate_answer"""
        return self.generate_answer(question, chunks)
    
    def invoke(self, question: str, context: str) -> str:
        """Direct invoke method"""
        try:
            response = self.client.models.generate_content(
                model=self.current_model,
                contents=f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"❌ Invoke error: {e}")
            return f"Error: {str(e)}"