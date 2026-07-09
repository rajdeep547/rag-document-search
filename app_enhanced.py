# app_enhanced.py
"""
Enhanced Document Q&A System - Streamlit Web Interface
Run with: streamlit run app_enhanced.py
"""
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Imports
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime
import tempfile
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend modules
from backend import DocumentQaOrchestrator
from backend.config import Config

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="📚 Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .doc-info {
        background: #d4edda;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        color: #155724;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .user-msg {
        background: #e3f2fd;
        padding: 0.7rem 1rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.4rem 0;
        max-width: 80%;
        float: right;
        clear: both;
        color: #1a1a1a;
    }
    
    .assistant-msg {
        background: #f0f0f0;
        padding: 0.7rem 1rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.4rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        color: #1a1a1a;
    }
    
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    
    .metric-card h3 {
        font-size: 0.8rem;
        color: #888;
        margin: 0;
    }
    
    .metric-card h2 {
        font-size: 2rem;
        color: #1f77b4;
        margin: 0.3rem 0 0 0;
    }
    
    .feedback-box {
        background: #fff3cd;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #ffc107;
        margin: 0.3rem 0;
    }
    
    .file-card {
        background: #f8f9fa;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.orchestrator = DocumentQaOrchestrator()
    st.session_state.chat_history = []
    st.session_state.documents = {}
    st.session_state.page = "💬 Chat"
    st.session_state.query_count = 0
    st.session_state.feedback_count = 0

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 0.5rem 0;'>
        <h2>📚 Document Q&A</h2>
        <p style='color: #666; font-size: 0.8rem;'>Gemini 1.5 Flash</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📊 Dashboard", "📁 Documents", "⭐ Feedback"],
        index=0
    )
    st.session_state.page = page
    
    st.divider()
    
    # Document Upload
    st.subheader("📤 Upload Documents")
    st.caption("Supported: PDF, DOCX, TXT, MD, CSV")
    
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=['pdf', 'docx', 'txt', 'md', 'csv'],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.documents]
        
        if new_files:
            for file in new_files:
                with st.spinner(f"⏳ Processing {file.name}..."):
                    try:
                        suffix = f".{file.name.split('.')[-1]}"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(file.getvalue())
                            tmp_path = tmp.name
                        
                        result = st.session_state.orchestrator.ingest_document(tmp_path)
                        
                        if result.get('success', False):
                            st.session_state.documents[file.name] = {
                                'size': f"{file.size / 1024:.1f} KB",
                                'chunks': result.get('chunks_created', 0),
                                'uploaded': datetime.now().strftime("%H:%M"),
                                'status': '✅'
                            }
                            st.success(f"✅ {file.name}")
                        else:
                            st.error(f"❌ {file.name}: {result.get('message', 'Error')}")
                            
                    except Exception as e:
                        st.error(f"❌ {file.name}: {str(e)[:50]}")
    
    st.divider()
    
    # Active Documents
    if st.session_state.documents:
        st.subheader(f"📚 Documents ({len(st.session_state.documents)})")
        
        for name, info in list(st.session_state.documents.items()):
            st.markdown(f"""
            <div class='file-card'>
                {info['status']} {name[:25]}{'...' if len(name) > 25 else ''}
                <span style='float:right;color:#999;font-size:0.8rem;'>{info['uploaded']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.orchestrator.clear_all()
            st.session_state.documents = {}
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    # System status
    try:
        status = st.session_state.orchestrator.get_status()
        st.caption(f"🟢 System Ready")
        st.caption(f"📚 Docs: {len(st.session_state.documents)}")
        st.caption(f"💬 Queries: {st.session_state.query_count}")
        st.caption(f"⚡ Cache: {'On' if status.get('cache_enabled') else 'Off'}")
        st.caption(f"🤖 Model: Gemini 1.5 Flash")
    except:
        st.caption("🟡 Initializing...")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.session_state.page == "💬 Chat":
    st.markdown("<h1 class='main-header'>💬 Chat with Your Documents</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload documents and ask questions naturally</p>", unsafe_allow_html=True)
    
    doc_count = len(st.session_state.documents)
    if doc_count > 0:
        st.markdown(f"<div class='doc-info'>✅ {doc_count} document(s) loaded. Ask a question below!</div>", unsafe_allow_html=True)
    else:
        st.info("📤 Upload documents from the sidebar to get started")
    
    st.markdown("---")
    
    # Chat display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.caption("💡 Ask a question about your documents...")
        
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; margin: 0.3rem 0;'>
                    <div class='user-msg'>
                        <strong>You</strong><br>{msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if msg.get('sources') and msg['sources']:
                    sources_html = f"<br><span style='font-size:0.8rem;color:#666;'>📎 Sources: {', '.join(msg['sources'][:2])}</span>"
                
                confidence_html = ""
                if msg.get('confidence') and msg['confidence'] != 'N/A' and msg['confidence'] != 'none':
                    confidence_html = f"<br><span style='font-size:0.8rem;color:#666;'>📊 Confidence: {msg['confidence']}</span>"
                
                cache_html = ""
                if msg.get('from_cache'):
                    cache_html = "<br><span style='font-size:0.8rem;color:#666;'>⚡ From cache</span>"
                
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-start; margin: 0.3rem 0;'>
                    <div class='assistant-msg'>
                        <strong>🤖 Assistant</strong><br>{msg['content']}
                        {sources_html}
                        {confidence_html}
                        {cache_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<div class='clearfix'></div>", unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        question = st.text_input(
            "Ask a question",
            placeholder="Type your question here...",
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_clicked = st.button("🚀 Ask", use_container_width=True, type="primary")
    
    with col3:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_clicked:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_clicked and question:
        if not st.session_state.documents:
            st.warning("⚠️ Please upload a document first!")
        else:
            st.session_state.chat_history.append({'role': 'user', 'content': question})
            
            with st.spinner("🤔 Thinking with Gemini..."):
                try:
                    result = st.session_state.orchestrator.ask(question)
                    st.session_state.query_count += 1
                    
                    if 'error' in result:
                        answer_text = f"❌ Error: {result['error']}"
                        sources = []
                        confidence = 'N/A'
                        from_cache = False
                    else:
                        answer_text = result.get('answer', 'No answer available')
                        sources = result.get('sources', [])
                        confidence = result.get('confidence', 'N/A')
                        from_cache = result.get('from_cache', False)
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer_text,
                        'sources': sources,
                        'confidence': confidence,
                        'from_cache': from_cache
                    })
                    
                except Exception as e:
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"❌ Error: {str(e)}",
                        'sources': [],
                        'confidence': 'N/A',
                        'from_cache': False
                    })
            
            st.rerun()

elif st.session_state.page == "📊 Dashboard":
    st.markdown("<h1 class='main-header'>📊 Dashboard</h1>", unsafe_allow_html=True)
    
    try:
        status = st.session_state.orchestrator.get_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>📄 Documents</h3>
                <h2>{len(st.session_state.documents)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            try:
                feedback_stats = st.session_state.orchestrator.get_feedback_stats()
                avg_rating = feedback_stats.get('average_rating', 0)
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>⭐ Avg Rating</h3>
                    <h2>{avg_rating:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>⭐ Avg Rating</h3>
                    <h2>0.0</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>💬 Total Queries</h3>
                <h2>{st.session_state.query_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            cache_hits = status.get('performance', {}).get('cache_hits', 0)
            st.markdown(f"""
            <div class='metric-card'>
                <h3>⚡ Cache Hits</h3>
                <h2>{cache_hits}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 System Status")
            st.json({
                'LLM Model': status.get('llm_model', 'N/A'),
                'Embedding Model': status.get('embedding_model', 'N/A'),
                'Cache': '✅ Enabled' if status.get('cache_enabled') else '❌ Disabled',
                'Documents': len(st.session_state.documents),
                'Uptime': status.get('uptime', 'N/A')
            })
        
        with col2:
            st.subheader("📈 Performance")
            perf = status.get('performance', {})
            st.metric("Avg Query Time", f"{perf.get('avg_query_time', 0):.2f}s")
            st.metric("Avg Ingestion Time", f"{perf.get('avg_ingestion_time', 0):.2f}s")
            st.metric("Total Queries", perf.get('total_queries', 0))
            
            total = perf.get('cache_hits', 0) + perf.get('cache_misses', 1)
            hit_ratio = (perf.get('cache_hits', 0) / total) * 100 if total > 0 else 0
            st.metric("Cache Hit Ratio", f"{hit_ratio:.1f}%")
        
        try:
            feedback_stats = st.session_state.orchestrator.get_feedback_stats()
            if feedback_stats.get('rating_distribution'):
                st.divider()
                st.subheader("⭐ Feedback Distribution")
                
                df = pd.DataFrame({
                    'Rating': list(feedback_stats['rating_distribution'].keys()),
                    'Count': list(feedback_stats['rating_distribution'].values())
                })
                df = df.sort_values('Rating')
                
                fig = px.bar(df, x='Rating', y='Count', title='', color='Rating', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        except:
            pass
        
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

elif st.session_state.page == "📁 Documents":
    st.markdown("<h1 class='main-header'>📁 Document Manager</h1>", unsafe_allow_html=True)
    
    if st.session_state.documents:
        doc_list = []
        for name, info in st.session_state.documents.items():
            doc_list.append({
                'Document': name,
                'Size': info.get('size', 'N/A'),
                'Chunks': info.get('chunks', 0),
                'Uploaded': info.get('uploaded', 'N/A'),
                'Status': info.get('status', '✅')
            })
        
        df = pd.DataFrame(doc_list)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Session Data", use_container_width=True):
                try:
                    st.session_state.orchestrator.export_session()
                    st.success("✅ Session exported to storage/exports/")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        with col2:
            if st.button("📊 View Details", use_container_width=True):
                for name, info in st.session_state.documents.items():
                    st.markdown(f"""
                    <div style='background:#f8f9fa; padding:0.6rem; border-radius:6px; margin:0.2rem 0;'>
                        <strong>{name}</strong><br>
                        Size: {info.get('size', 'N/A')} | 
                        Chunks: {info.get('chunks', 0)} | 
                        Uploaded: {info.get('uploaded', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("📚 No documents uploaded yet. Upload documents from the sidebar.")

elif st.session_state.page == "⭐ Feedback":
    st.markdown("<h1 class='main-header'>⭐ Feedback & Analytics</h1>", unsafe_allow_html=True)
    
    try:
        feedback_stats = st.session_state.orchestrator.get_feedback_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Feedback", feedback_stats.get('total_feedback', 0))
        with col2:
            st.metric("Average Rating", f"{feedback_stats.get('average_rating', 0):.1f} ⭐")
        with col3:
            st.metric("Unique Questions", feedback_stats.get('unique_questions', 0))
        
        st.divider()
        
        st.subheader("📝 Recent Feedback")
        recent = feedback_stats.get('recent_feedback', [])
        
        if recent:
            for fb in recent[:5]:
                st.markdown(f"""
                <div class='feedback-box'>
                    <strong>Q:</strong> {fb.get('question', '')[:100]}<br>
                    <strong>Rating:</strong> {'⭐' * fb.get('rating', 0)}<br>
                    <strong>When:</strong> {fb.get('timestamp', '')[:16]}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No feedback collected yet.")
        
        st.divider()
        st.subheader("💡 Improvement Suggestions")
        try:
            suggestions = st.session_state.orchestrator.get_improvement_suggestions()
            for suggestion in suggestions:
                st.info(suggestion)
        except:
            st.info("Collect more feedback for insights")
            
    except Exception as e:
        st.error(f"Error loading feedback: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(f"""
<div style='text-align: center; color: #999; font-size: 0.75rem; padding: 0.5rem;'>
    🚀 Built with LangChain · Gemini 1.5 Flash · ChromaDB
    <br>
    📚 {len(st.session_state.documents)} docs · 💬 {st.session_state.query_count} queries · ⭐ {st.session_state.feedback_count} feedbacks
</div>
""", unsafe_allow_html=True)