import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Voice Cloner Pro - Limited Cloud Version",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .feature-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🎙️ Advanced Voice Cloner</h1>", unsafe_allow_html=True)

# Info box
st.markdown("""
<div class='info-box'>
    <h2>Cloud Version Limitations</h2>
    <p>Due to Streamlit Cloud's resource constraints, this is a limited version of Voice Cloner Pro.</p>
    <p>For full functionality with voice cloning capabilities, please run the application locally from our GitHub repository.</p>
</div>
""", unsafe_allow_html=True)

# Features
st.markdown("<h2 class='feature-header'>Application Features</h2>", unsafe_allow_html=True)

features = [
    "🔊 Voice Cloning from Audio Samples",
    "🌐 YouTube Link Support",
    "🎭 Accent and Prosody Transfer",
    "⚙️ Advanced Voice Tuning",
    "📝 Text-to-Speech with Cloned Voice",
    "🔄 Iterative Voice Refinement"
]

# Convert features to DataFrame for display
df = pd.DataFrame(features, columns=["Feature"])
st.table(df)

# GitHub link
st.markdown("""
<div style='text-align: center; margin: 30px;'>
    <a href="https://github.com/premsingh23/Voice-Cloner--Live-LLM-using-GPT" 
       target="_blank" style="background-color: #24292e; color: white; padding: 12px 24px; 
       text-decoration: none; border-radius: 4px; font-weight: bold;">
        Get Full Version on GitHub
    </a>
</div>
""", unsafe_allow_html=True)

# System Status
st.markdown("<h2 class='feature-header'>System Status</h2>", unsafe_allow_html=True)
status_col1, status_col2 = st.columns(2)

with status_col1:
    st.metric(label="App Version", value="Cloud 1.0")
    st.metric(label="Last Updated", value=datetime.now().strftime("%Y-%m-%d"))

with status_col2:
    st.metric(label="Status", value="Active")
    st.metric(label="Available Memory", value="Limited")

# Footer
st.markdown("---")
st.markdown("© 2024 Voice Cloner Pro | Made with ❤️ using Streamlit") 