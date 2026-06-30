"""
TeamMind - Complete Agent-Integrated Streamlit Frontend
Full integration with all 6 specialized agents
"""
import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import os
import re

# Page configuration
st.set_page_config(
    page_title="TeamMind - Agent Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    /* Agent Cards */
    .agent-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        background: #f8f9fa;
    }
    
    .agent-card-summary { border-left-color: #4ECDC4; }
    .agent-card-decision { border-left-color: #FF6B6B; }
    .agent-card-risk { border-left-color: #FFA94D; }
    .agent-card-task { border-left-color: #45B7D1; }
    .agent-card-signal { border-left-color: #96CEB4; }
    .agent-card-graph { border-left-color: #DDA0DD; }
    
    /* Badges */
    .badge-confidence-high { 
        background: #28a745; 
        color: white; 
        padding: 0.2rem 0.5rem; 
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .badge-confidence-medium { 
        background: #ffc107; 
        color: black; 
        padding: 0.2rem 0.5rem; 
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .badge-confidence-low { 
        background: #dc3545; 
        color: white; 
        padding: 0.2rem 0.5rem; 
        border-radius: 10px;
        font-size: 0.8rem;
    }
    
    .decision-approved { color: #28a745; font-weight: bold; }
    .decision-rejected { color: #dc3545; font-weight: bold; }
    .decision-postponed { color: #ffc107; font-weight: bold; }
    .decision-recorded { color: #6c757d; font-weight: bold; }
    
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    
    .task-pending { color: #ffc107; }
    .task-in-progress { color: #45B7D1; }
    .task-completed { color: #28a745; }
    
    /* Source items */
    .source-item {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        border-left: 3px solid #28a745;
    }
    
    .source-score {
        color: #667eea;
        font-weight: bold;
    }
    
    /* Modality badges */
    .modality-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.1rem;
    }
    .badge-audio { background: #FF6B6B; color: white; }
    .badge-document { background: #4ECDC4; color: white; }
    .badge-image { background: #45B7D1; color: white; }
    .badge-slide { background: #96CEB4; color: white; }
    .badge-email { background: #DDA0DD; color: white; }
    .badge-chat { background: #F0E68C; color: #333; }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .card { background: #2d2d2d; color: white; }
        .subtitle { color: #aaa; }
        .agent-card { background: #2d2d2d; }
        .source-item { background: #2d2d2d; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS (Defined BEFORE they are used)
# ============================================================================

def display_summary_content(content):
    """Display summary agent output"""
    if isinstance(content, dict):
        # Full summary
        if content.get('full_summary'):
            st.markdown("#### 📝 Full Summary")
            st.write(content['full_summary'])
        
        # Sections
        if content.get('sections'):
            st.markdown("#### 📑 Sections")
            for section, text in content['sections'].items():
                if text:
                    st.markdown(f"**{section.capitalize()}:** {text}")
        
        # Key points
        if content.get('key_points'):
            st.markdown("#### 🎯 Key Points")
            for point in content['key_points']:
                st.markdown(f"- {point}")


def display_decision_content(content):
    """Display decision agent output"""
    if isinstance(content, list):
        for decision in content:
            decision_type = decision.get('type', 'recorded')
            type_class = {
                'approved': 'decision-approved',
                'rejected': 'decision-rejected',
                'postponed': 'decision-postponed',
                'recorded': 'decision-recorded'
            }.get(decision_type, 'decision-recorded')
            
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f8f9fa; border-radius: 5px;">
                <span class="{type_class}">● {decision_type.upper()}</span>
                <p style="margin: 0.3rem 0;">{decision.get('text', '')}</p>
                <small>Confidence: {decision.get('confidence', 0):.1%}</small>
            </div>
            """, unsafe_allow_html=True)


def display_risk_content(content):
    """Display risk agent output"""
    if isinstance(content, dict):
        for category, risks in content.items():
            if risks:
                st.markdown(f"#### {category.capitalize()} Risks")
                for risk in risks:
                    severity = risk.get('severity', 'low')
                    severity_class = f"risk-{severity}"
                    st.markdown(f"""
                    <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f8f9fa; border-radius: 5px;">
                        <span class="{severity_class}">● {severity.upper()}</span>
                        <p style="margin: 0.3rem 0;">{risk.get('text', '')}</p>
                        <small>Confidence: {risk.get('confidence', 0):.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)


def display_task_content(content):
    """Display task agent output"""
    if isinstance(content, list):
        for task in content:
            status = task.get('status', 'pending')
            status_class = f"task-{status.replace('_', '-')}"
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f8f9fa; border-radius: 5px;">
                <strong>{task.get('title', 'Task')}</strong>
                <span class="{status_class}">● {status.upper()}</span>
                <p style="margin: 0.3rem 0;">{task.get('description', '')}</p>
                <small>
                    👤 Owner: {task.get('owner', 'Unassigned')} | 
                    📅 Deadline: {task.get('deadline', 'Not specified')} |
                    Priority: {task.get('priority', 'medium')}
                </small>
            </div>
            """, unsafe_allow_html=True)


def display_signal_content(content):
    """Display weak signal agent output"""
    if isinstance(content, list):
        for signal in content:
            strength = signal.get('strength', 0)
            strength_indicator = "🟢" if strength > 0.7 else "🟡" if strength > 0.4 else "🔴"
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f8f9fa; border-radius: 5px;">
                <strong>{signal.get('category', 'Signal').capitalize()}</strong>
                {strength_indicator} Strength: {strength:.1%}
                <p style="margin: 0.3rem 0;">{signal.get('text', '')}</p>
                <small>Reasoning: {signal.get('reasoning', 'N/A')}</small>
            </div>
            """, unsafe_allow_html=True)


def display_graph_content(content):
    """Display knowledge graph agent output"""
    if isinstance(content, dict):
        # Entities
        if content.get('entities'):
            st.markdown("#### 🔷 Entities")
            entities_text = ", ".join([e.get('name', '') for e in content['entities'][:10]])
            st.write(entities_text)
            if len(content['entities']) > 10:
                st.caption(f"+ {len(content['entities']) - 10} more entities")
        
        # Relationships
        if content.get('relationships'):
            st.markdown("#### 🔗 Relationships")
            for rel in content['relationships'][:5]:
                st.markdown(f"- {rel.get('source', '')} → {rel.get('type', '')} → {rel.get('target', '')}")
            if len(content['relationships']) > 5:
                st.caption(f"+ {len(content['relationships']) - 5} more relationships")
        
        # Insights
        if content.get('insights'):
            st.markdown("#### 💡 Insights")
            for insight in content['insights']:
                st.markdown(f"- **{insight.get('entity', '')}**: {insight.get('type', '')} (Degree: {insight.get('degree', 0)})")


def display_agent_output(agent_name: str, agent_output: dict):
    """
    Display individual agent output with proper formatting
    """
    # Map agent names to icons and colors
    agent_info = {
        "summary": {"icon": "📝", "label": "Summary Agent", "color": "#4ECDC4", "card_class": "agent-card-summary"},
        "decision": {"icon": "📊", "label": "Decision Agent", "color": "#FF6B6B", "card_class": "agent-card-decision"},
        "risk": {"icon": "⚠️", "label": "Risk Agent", "color": "#FFA94D", "card_class": "agent-card-risk"},
        "task": {"icon": "📋", "label": "Task Agent", "color": "#45B7D1", "card_class": "agent-card-task"},
        "weak_signal": {"icon": "📡", "label": "Weak Signal Agent", "color": "#96CEB4", "card_class": "agent-card-signal"},
        "knowledge_graph": {"icon": "🕸️", "label": "Knowledge Graph Agent", "color": "#DDA0DD", "card_class": "agent-card-graph"}
    }
    
    info = agent_info.get(agent_name, {"icon": "🤖", "label": agent_name, "color": "#667eea", "card_class": ""})
    
    with st.expander(f"{info['icon']} {info['label']}", expanded=False):
        # Display confidence if available
        confidence = agent_output.get('confidence', 0) if isinstance(agent_output, dict) else 0
        if confidence > 0:
            if confidence > 0.7:
                badge_class = "badge-confidence-high"
            elif confidence > 0.4:
                badge_class = "badge-confidence-medium"
            else:
                badge_class = "badge-confidence-low"
            st.markdown(f'<span class="{badge_class}">Confidence: {confidence:.1%}</span>', unsafe_allow_html=True)
        
        # Check for error
        if isinstance(agent_output, dict) and 'error' in agent_output:
            st.error(f"Error: {agent_output['error']}")
            return
        
        # Display content based on agent type
        content = agent_output.get('content', {}) if isinstance(agent_output, dict) else agent_output
        
        if not content:
            st.info("No content available from this agent")
            return
        
        if agent_name == "summary":
            display_summary_content(content)
        elif agent_name == "decision":
            display_decision_content(content)
        elif agent_name == "risk":
            display_risk_content(content)
        elif agent_name == "task":
            display_task_content(content)
        elif agent_name == "weak_signal":
            display_signal_content(content)
        elif agent_name == "knowledge_graph":
            display_graph_content(content)
        else:
            # Fallback: show as JSON
            st.json(content)


# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

# Initialize session state - all at the top
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = "All Agents"
if 'multimodal_stats' not in st.session_state:
    st.session_state.multimodal_stats = {
        'audio': 0,
        'documents': 0,
        'images': 0,
        'slides': 0,
        'emails': 0,
        'chats': 0
    }
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""
if 'last_upload_result' not in st.session_state:
    st.session_state.last_upload_result = None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.markdown("### 🧠 TeamMind")
    st.markdown("*Multi-Agent Intelligence*")
    st.markdown("---")
    
    # API Configuration
    st.markdown("### ⚙️ Configuration")
    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="URL of the TeamMind API server"
    )
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
        st.rerun()
    
    # Check API connection
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ Connected to API")
            data = response.json()
            st.info(f"🤖 LLM: {data.get('model', 'Groq')}")
            st.info(f"📊 Vectors: {data.get('vector_store_points', 0)}")
            
            # Get agents info
            try:
                agents_response = requests.get(f"{st.session_state.api_url}/agents", timeout=2)
                if agents_response.status_code == 200:
                    agents_data = agents_response.json()
                    st.success(f"🤖 {len(agents_data.get('agents', []))} Agents Active")
            except:
                pass
        else:
            st.error("❌ API not responding")
    except:
        st.error("❌ Cannot connect to API")
        st.info("Make sure the API server is running")
    
    st.markdown("---")
    
    # Agent Selection
    st.markdown("### 🎯 Agent Focus")
    agent_options = [
        "All Agents",
        "📝 Summary Agent",
        "📊 Decision Agent",
        "⚠️ Risk Agent",
        "📋 Task Agent",
        "📡 Weak Signal Agent",
        "🕸️ Knowledge Graph Agent"
    ]
    
    selected_agent = st.selectbox(
        "Focus on specific agent",
        agent_options,
        index=0
    )
    st.session_state.selected_agent = selected_agent
    
    st.markdown("---")
    
    # Quick Filters
    st.markdown("### 🔍 Quick Filters")
    department = st.selectbox(
        "Department",
        ["All", "Engineering", "Marketing", "Sales", "Finance", "HR", "Operations", "Legal", "R&D"]
    )
    
    top_k = st.slider(
        "Number of Sources",
        min_value=1,
        max_value=15,
        value=5,
        help="How many source documents to retrieve"
    )
    
    st.markdown("---")
    
    # Sample Queries by Agent Type
    st.markdown("### 💡 Agent-Specific Queries")
    
    agent_queries = {
        "📝 Summary": "Summarize the quarterly review meeting",
        "📊 Decisions": "What decisions were made about hiring?",
        "⚠️ Risks": "What are the main risks discussed?",
        "📋 Tasks": "What action items were assigned?",
        "📡 Signals": "What are the emerging patterns or concerns?",
        "🕸️ Graph": "Show relationships between people and projects",
        "🎯 All": "Give me a complete analysis of the meeting"
    }
    
    for label, query in agent_queries.items():
        if st.button(label, use_container_width=True):
            st.session_state.query_text = query
            st.rerun()
    
    st.markdown("---")
    
    # Upload Section
    st.markdown("### 📤 Upload Files")
    uploaded_file = st.file_uploader(
        "Upload meeting files",
        type=['mp3', 'wav', 'pdf', 'docx', 'pptx', 'txt', 'png', 'jpg', 'jpeg', 'md'],
        help="Upload audio recordings, documents, or images"
    )
    
    if uploaded_file:
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        # Determine modality
        if file_ext in ['.mp3', '.wav', '.m4a', '.flac']:
            modality = "audio"
            modality_icon = "🎙️"
        elif file_ext in ['.pdf', '.docx', '.txt', '.md']:
            modality = "document"
            modality_icon = "📄"
        elif file_ext in ['.pptx', '.ppt']:
            modality = "slide"
            modality_icon = "📊"
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            modality = "image"
            modality_icon = "🖼️"
        elif file_ext in ['.msg', '.eml']:
            modality = "email"
            modality_icon = "📧"
        else:
            modality = "document"
            modality_icon = "📄"
        
        file_metadata = {
            "filename": uploaded_file.name,
            "file_type": uploaded_file.type,
            "size": len(uploaded_file.getvalue()),
            "uploaded_at": datetime.now().isoformat(),
            "modality": modality,
            "modality_icon": modality_icon
        }
        
        if st.button(f"📤 Upload {modality_icon} {uploaded_file.name}", type="primary", use_container_width=True):
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    metadata_json = json.dumps({
                        "modality": modality,
                        "department": department if department != "All" else "",
                        "uploaded_by": "streamlit_user",
                        "date": datetime.now().isoformat()
                    })
                    data = {'metadata': metadata_json}
                    
                    response = requests.post(
                        f"{st.session_state.api_url}/upload",
                        files=files,
                        data=data,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                        
                        # Show storage info
                        chunks_stored = result.get('chunks_stored', 0)
                        total_chunks = result.get('total_chunks_extracted', 0)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Chunks Stored", chunks_stored)
                        with col2:
                            st.metric("Extracted", total_chunks)
                        
                        if chunks_stored == 0:
                            st.warning("⚠️ No text chunks were stored. Make sure the file contains readable text.")
                        else:
                            st.session_state.multimodal_stats[modality] = st.session_state.multimodal_stats.get(modality, 0) + 1
                            st.session_state.uploaded_files.append(file_metadata)
                            st.session_state.last_upload_result = result
                        
                        # Show extracted content preview
                        if result.get('result') and result['result'].get('chunks'):
                            with st.expander("📄 Extracted Content Preview"):
                                for i, chunk in enumerate(result['result']['chunks'][:3]):
                                    st.write(f"**Chunk {i+1}:**")
                                    st.write(chunk.get('content', '')[:300] + '...')
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<p class="main-title">🧠 TeamMind</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-Agent Meeting Intelligence System powered by Groq LLM</p>', unsafe_allow_html=True)

# Agent Quick Stats
cols = st.columns(6)
agent_icons = ["📝", "📊", "⚠️", "📋", "📡", "🕸️"]
agent_names = ["Summary", "Decision", "Risk", "Task", "Signal", "Graph"]

for col, icon, name in zip(cols, agent_icons, agent_names):
    with col:
        st.markdown(f"""
        <div style='text-align: center; padding: 0.5rem; background: #f8f9fa; border-radius: 8px;'>
            <div style='font-size: 1.5rem;'>{icon}</div>
            <div style='font-weight: bold;'>{name}</div>
            <div style='font-size: 0.8rem; color: #666;'>Agent</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Query & Agents",
    "🎯 Multimodal Analysis",
    "📊 Dashboard",
    "📁 Files",
    "📚 History"
])

# ============================================================================
# TAB 1: Query & Agents
# ============================================================================

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_text = st.text_area(
            "Ask a question about your meetings",
            value=st.session_state.query_text,
            height=80,
            placeholder="E.g., What decisions were made about the budget? What are the risks?"
        )
    
    with col2:
        st.write("")
        st.write("")
        query_button = st.button("🔍 Search", type="primary", use_container_width=True)
        clear_button = st.button("🗑️ Clear", use_container_width=True)
        if clear_button:
            st.session_state.query_text = ""
            st.session_state.current_response = None
            st.rerun()
    
    # Process query
    if query_button and query_text:
        st.session_state.processing = True
        
        with st.spinner("🧠 Processing with all agents..."):
            try:
                payload = {
                    "query": query_text,
                    "top_k": top_k
                }
                
                if department != "All":
                    payload["filter_conditions"] = {"department": department}
                
                response = requests.post(
                    f"{st.session_state.api_url}/query",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.current_response = data
                    st.session_state.query_history.append({
                        "query": query_text,
                        "timestamp": datetime.now().isoformat(),
                        "response": data
                    })
                    st.rerun()
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.session_state.processing = False
    
    # Display Response with Agent Integration
    if st.session_state.current_response:
        data = st.session_state.current_response
        
        # Overall Answer
        st.markdown("### 📝 Overall Answer")
        st.markdown(f'<div class="card">{data["answer"]}</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Confidence", f"{data['confidence']:.1%}")
        with col2:
            st.metric("Sources", len(data['sources']))
        with col3:
            st.metric("Agents Used", len(data['agent_outputs']))
        with col4:
            st.metric("LLM", "Groq")
        
        # Agent Outputs - Full Integration
        if data.get('agent_outputs'):
            st.markdown("### 🤖 Agent Analysis")
            
            # Filter agents based on selection
            agent_outputs = data['agent_outputs']
            selected_agent = st.session_state.selected_agent
            
            if selected_agent != "All Agents":
                # Map selection to agent key
                agent_map = {
                    "📝 Summary Agent": "summary",
                    "📊 Decision Agent": "decision",
                    "⚠️ Risk Agent": "risk",
                    "📋 Task Agent": "task",
                    "📡 Weak Signal Agent": "weak_signal",
                    "🕸️ Knowledge Graph Agent": "knowledge_graph"
                }
                agent_key = agent_map.get(selected_agent)
                if agent_key and agent_key in agent_outputs:
                    agent_outputs = {agent_key: agent_outputs[agent_key]}
                else:
                    st.warning(f"Agent {selected_agent} output not found")
            
            # Display each agent's output
            for agent_name, agent_output in agent_outputs.items():
                display_agent_output(agent_name, agent_output)
        
        # Sources
        if data.get('sources'):
            st.markdown("### 📚 Sources")
            for i, source in enumerate(data['sources'][:5]):
                with st.expander(f"Source {i+1}: {source.get('source_file', 'Unknown')}"):
                    st.markdown(f"**Score:** {source.get('score', 0):.3f}")
                    st.markdown(f"**Text:** {source.get('text', '')[:500]}...")
                    if source.get('metadata'):
                        st.markdown("**Metadata:**")
                        st.json(source['metadata'])

# ============================================================================
# TAB 2: Multimodal Analysis
# ============================================================================

with tab2:
    st.markdown("### 🎯 Multimodal Analysis")
    
    if st.session_state.uploaded_files:
        # Create a visualization of file types
        file_types = [f.get('modality', 'unknown') for f in st.session_state.uploaded_files]
        
        if file_types:
            type_counts = pd.Series(file_types).value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.3,
                marker=dict(
                    colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD', '#F0E68C']
                )
            )])
            fig.update_layout(
                title="File Distribution by Modality",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed file list
        st.markdown("### 📋 File Details")
        
        files_df = pd.DataFrame(st.session_state.uploaded_files)
        if 'modality' in files_df.columns:
            modality_icons = {
                'audio': '🎙️',
                'document': '📄',
                'image': '🖼️',
                'slide': '📊',
                'email': '📧',
                'chat': '💬'
            }
            files_df['icon'] = files_df['modality'].map(modality_icons)
            
            st.dataframe(
                files_df[['icon', 'filename', 'modality', 'uploaded_at']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'icon': st.column_config.TextColumn('', width='small'),
                    'filename': st.column_config.TextColumn('Filename'),
                    'modality': st.column_config.TextColumn('Modality'),
                    'uploaded_at': st.column_config.DatetimeColumn('Uploaded At')
                }
            )
    else:
        st.info("No files uploaded yet. Upload files from the sidebar.")

# ============================================================================
# TAB 3: Dashboard
# ============================================================================

with tab3:
    st.markdown("### 📊 System Dashboard")
    
    # System Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
            if response.status_code == 200:
                st.metric("API Status", "🟢 Online")
            else:
                st.metric("API Status", "🔴 Offline")
        except:
            st.metric("API Status", "🔴 Offline")
    
    with col2:
        st.metric("Total Files", len(st.session_state.uploaded_files))
    
    with col3:
        st.metric("Total Queries", len(st.session_state.query_history))
    
    with col4:
        st.metric("Active Agents", 6)
    
    # Multimodal Stats
    st.markdown("### 📊 Multimodal Statistics")
    
    modalities = ['audio', 'documents', 'images', 'slides', 'emails', 'chats']
    counts = [st.session_state.multimodal_stats.get(m, 0) for m in modalities]
    labels = ['🎙️ Audio', '📄 Documents', '🖼️ Images', '📊 Slides', '📧 Emails', '💬 Chats']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=counts,
            text=counts,
            textposition='auto',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD', '#F0E68C']
        )
    ])
    fig.update_layout(
        title="Files by Modality",
        xaxis_title="Modality",
        yaxis_title="Count",
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Agent Performance Metrics
    if st.session_state.query_history:
        st.markdown("### 📊 Agent Performance")
        
        metrics_data = []
        for entry in st.session_state.query_history[-10:]:
            if 'response' in entry and 'agent_outputs' in entry['response']:
                agent_outputs = entry['response']['agent_outputs']
                for agent_name, output in agent_outputs.items():
                    confidence = output.get('confidence', 0) if isinstance(output, dict) else 0
                    metrics_data.append({
                        "Agent": agent_name,
                        "Confidence": confidence,
                        "Timestamp": entry.get('timestamp', '')
                    })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            if not df.empty:
                agent_metrics = df.groupby('Agent')['Confidence'].mean().sort_values(ascending=False)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=agent_metrics.index,
                        y=agent_metrics.values,
                        text=[f"{v:.1%}" for v in agent_metrics.values],
                        textposition='auto',
                        marker_color=['#4ECDC4', '#FF6B6B', '#FFA94D', '#45B7D1', '#96CEB4', '#DDA0DD']
                    )
                ])
                fig.update_layout(
                    title="Average Confidence by Agent",
                    xaxis_title="Agent",
                    yaxis_title="Confidence Score",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: Files
# ============================================================================

with tab4:
    st.markdown("### 📁 Uploaded Files")
    
    if st.session_state.uploaded_files:
        files_df = pd.DataFrame(st.session_state.uploaded_files)
        
        if 'modality' in files_df.columns:
            modality_icons = {
                'audio': '🎙️',
                'document': '📄',
                'image': '🖼️',
                'slide': '📊',
                'email': '📧',
                'chat': '💬'
            }
            files_df['icon'] = files_df['modality'].map(modality_icons)
            
            st.dataframe(
                files_df[['icon', 'filename', 'modality', 'uploaded_at']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'icon': st.column_config.TextColumn('', width='small'),
                    'filename': st.column_config.TextColumn('Filename'),
                    'modality': st.column_config.TextColumn('Modality'),
                    'uploaded_at': st.column_config.DatetimeColumn('Uploaded At')
                }
            )
    else:
        st.info("No files uploaded yet. Upload files from the sidebar.")

# ============================================================================
# TAB 5: History
# ============================================================================

with tab5:
    st.markdown("### 📚 Query History with Agent Outputs")
    
    if st.session_state.query_history:
        for entry in reversed(st.session_state.query_history[-20:]):
            with st.expander(f"🔍 {entry['query'][:50]}... - {entry['timestamp'][:19]}"):
                st.markdown(f"**Query:** {entry['query']}")
                
                if 'response' in entry:
                    response = entry['response']
                    
                    if 'agent_outputs' in response:
                        agents_used = list(response['agent_outputs'].keys())
                        st.markdown(f"**Agents Used:** {', '.join(agents_used)}")
                    
                    st.markdown("**Answer Preview:**")
                    st.write(response.get('answer', '')[:200] + '...')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{response.get('confidence', 0):.1%}")
                    with col2:
                        st.metric("Sources", len(response.get('sources', [])))
    else:
        st.info("No query history yet. Start asking questions!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.caption("🧠 TeamMind v2.0 - Agent Integration")
with col2:
    st.caption("⚡ Powered by Groq LLM")
with col3:
    st.caption("🤖 6 Specialized Agents")
with col4:
    st.caption("🎯 Full Agent Orchestration")

# ============================================================================
# NO AUTO-REFRESH - Removed the infinite loop
# ============================================================================
# The auto-refresh has been removed to prevent infinite loading