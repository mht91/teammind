"""
Multimodal helper functions for TeamMind
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import base64
import json
from PIL import Image
import io

def display_file_preview(file_data, modality):
    """Display file preview based on modality"""
    if modality == 'image':
        try:
            # Decode base64 image
            image_data = base64.b64decode(file_data['data'])
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption=file_data.get('filename', 'Image'), use_container_width=True)
        except:
            st.warning("Image preview not available")
    
    elif modality == 'audio':
        st.audio(file_data['data'], format='audio/mp3')
        st.caption(f"Audio file: {file_data.get('filename', 'Unknown')}")
    
    elif modality == 'document':
        if file_data.get('content'):
            with st.expander("📄 Document Preview"):
                st.write(file_data['content'][:500] + '...')
    
    elif modality == 'slide':
        st.info("📊 Slide content available for analysis")
        if file_data.get('content'):
            with st.expander("📊 Slide Preview"):
                st.write(file_data['content'][:300] + '...')
    
    elif modality == 'email':
        st.info("📧 Email content available for analysis")
        if file_data.get('content'):
            with st.expander("📧 Email Preview"):
                st.write(file_data['content'][:300] + '...')
    
    elif modality == 'chat':
        st.info("💬 Chat content available for analysis")
        if file_data.get('content'):
            with st.expander("💬 Chat Preview"):
                st.write(file_data['content'][:300] + '...')

def create_modality_distribution_chart(files):
    """Create a pie chart showing modality distribution"""
    if not files:
        return None
    
    modality_counts = {}
    for f in files:
        mod = f.get('modality', 'unknown')
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
    
    fig = go.Figure(data=[go.Pie(
        labels=list(modality_counts.keys()),
        values=list(modality_counts.values()),
        hole=0.3,
        marker=dict(
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD', '#F0E68C']
        )
    )])
    fig.update_layout(
        title="Modality Distribution",
        height=300,
        showlegend=True
    )
    return fig

def get_modality_icon(modality):
    """Get icon for modality"""
    icons = {
        'audio': '🎙️',
        'document': '📄',
        'image': '🖼️',
        'slide': '📊',
        'email': '📧',
        'chat': '💬'
    }
    return icons.get(modality, '📄')

def get_modality_color(modality):
    """Get color for modality"""
    colors = {
        'audio': '#FF6B6B',
        'document': '#4ECDC4',
        'image': '#45B7D1',
        'slide': '#96CEB4',
        'email': '#DDA0DD',
        'chat': '#F0E68C'
    }
    return colors.get(modality, '#667eea')