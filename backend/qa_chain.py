# backend/qa_chain.py
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import threading
import os
from dotenv import load_dotenv

load_dotenv()

class QaChain:
    # ✅ Default model should be gemini-3.5-flash
    def __init__(self, model="gemini-3.5-flash", temperature=0):
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,  # This will be gemini-3.5-flash
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,
            request_timeout=55
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful document assistant. Answer based ONLY on the context.

CONTEXT:
{context}

INSTRUCTIONS:
1. Answer ONLY using the context above
2. If the context doesn't have the answer, say: "I cannot find this information."
3. Be concise (1-2 sentences)

ANSWER:"""),
            ("human", "{question}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.request_count = 0
        self.last_request_time = 0
        print(f"✅ QaChain ready with {model}")