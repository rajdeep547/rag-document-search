# backend/qa_chain.py
"""
QA Chain using the Groq SDK
"""
from typing import List, Dict, Optional
import time
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class QaChain:
    """
    QA Chain using Groq-hosted Llama models
    """

    def __init__(self, model="llama-3.3-70b-versatile", temperature=0):
        """
        Initialize QA Chain

        Args:
            model: Groq model name (default: llama-3.3-70b-versatile)
            temperature: Temperature for generation (0 = factual)
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.api_key = api_key
        self.temperature = temperature

        # Primary model and fallbacks
        self.models = [
            "llama-3.3-70b-versatile",  # Primary - Capable and fast
            "llama-3.1-8b-instant",     # Fallback 1 - Fastest
            "gemma2-9b-it"               # Fallback 2 - Alternative
        ]
        if model not in self.models:
            self.models.insert(0, model)
        self.current_model_index = self.models.index(model)
        self.current_model = model

        # Initialize the client
        self.client = Groq(api_key=api_key)

        self.request_count = 0
        self.last_request_time = 0
        print(f"✅ QaChain ready with {self.current_model}")
        print(f"   🔄 Fallback models: {', '.join(m for m in self.models if m != self.current_model)}")

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

                # Generate answer using the Groq SDK
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=2048,
                )

                answer = response.choices[0].message.content.strip()

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

                # Check if it's a model not found / decommissioned error
                if "404" in error_msg or "not_found" in error_msg.lower() or "decommissioned" in error_msg.lower():
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
                elif "503" in error_msg or "rate_limit" in error_msg.lower() or "high demand" in error_msg.lower():
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

    def generate_summary(self, chunks: List[Dict], doc_name: str = "the document", length: str = "medium") -> Dict:
        """
        Generate a summary of a document from its chunks.

        Uses map-reduce (summarize batches, then summarize the summaries)
        for documents too large to fit in a single prompt.

        Args:
            chunks: All chunks belonging to the document, in reading order
            doc_name: Display name of the document
            length: "short", "medium", or "long"

        Returns:
            Dictionary with summary, generation_time, chunks_used
        """
        start_time = time.time()

        if not chunks:
            return {
                "summary": "No content available to summarize.",
                "generation_time": 0,
                "chunks_used": 0
            }

        MAX_CHARS_PER_BATCH = 12000
        batches = self._batch_chunks(chunks, MAX_CHARS_PER_BATCH)

        try:
            if len(batches) == 1:
                summary = self._summarize_text(self._join_chunks(batches[0]), doc_name, length)
            else:
                partial_summaries = []
                for i, batch in enumerate(batches, 1):
                    print(f"   📝 Summarizing part {i}/{len(batches)}...")
                    partial = self._summarize_text(self._join_chunks(batch), doc_name, "concise")
                    partial_summaries.append(partial)

                combined = "\n\n".join(partial_summaries)
                summary = self._summarize_text(combined, doc_name, length, is_reduce=True)

            return {
                "summary": summary,
                "generation_time": round(time.time() - start_time, 2),
                "chunks_used": len(chunks),
                "parts_summarized": len(batches),
                "model_used": self.current_model
            }
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)[:200]}",
                "generation_time": round(time.time() - start_time, 2),
                "chunks_used": len(chunks),
                "error": str(e)
            }

    def _batch_chunks(self, chunks: List[Dict], max_chars: int) -> List[List[Dict]]:
        """Group chunks into batches that stay under a character budget"""
        batches = []
        current: List[Dict] = []
        current_len = 0
        for chunk in chunks:
            content = chunk.get('content', '')
            if current and current_len + len(content) > max_chars:
                batches.append(current)
                current = []
                current_len = 0
            current.append(chunk)
            current_len += len(content)
        if current:
            batches.append(current)
        return batches

    def _join_chunks(self, chunks: List[Dict]) -> str:
        return "\n\n".join(c.get('content', '') for c in chunks if c.get('content'))

    def _summarize_text(self, text: str, doc_name: str, length: str, is_reduce: bool = False) -> str:
        length_instructions = {
            "short": "in 2-3 sentences",
            "medium": "in 1-2 short paragraphs",
            "concise": "in 3-4 sentences, keeping only the most important points",
            "long": "in detail, covering all major points, using multiple paragraphs"
        }
        instruction = length_instructions.get(length, length_instructions["medium"])

        if is_reduce:
            prompt = f"""The following are summaries of different sections of a document called "{doc_name}". Combine them into one coherent summary {instruction}. Do not mention that it was created from separate sections.

SECTION SUMMARIES:
{text}

FINAL SUMMARY:"""
        else:
            prompt = f"""Summarize the following document content {instruction}. Focus on the key points, main ideas, and important details.

DOCUMENT: {doc_name}

CONTENT:
{text}

SUMMARY:"""

        response = self.client.chat.completions.create(
            model=self.current_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

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
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[{
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                }],
                temperature=self.temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Invoke error: {e}")
            return f"Error: {str(e)}"
