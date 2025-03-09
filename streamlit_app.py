import os
import streamlit as st
import logging
import time
import tempfile
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_cloner_app")

# Set environment variable for CPU-only operation on Streamlit Cloud
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Check if running on Streamlit Cloud
is_streamlit_cloud = os.environ.get('STREAMLIT_SHARING', '') == 'true' or os.environ.get('STREAMLIT_CLOUD', '') == 'true'

# Initialize session state variables
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

if 'voice_characteristics' not in st.session_state:
    st.session_state.voice_characteristics = {
        'has_analysis': False,
        'speaking_rate': None,
        'pitch_mean': None,
        'pitch_range': None
    }

def main():
    st.set_page_config(
        page_title="Voice Cloner Pro - YourTTS",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Apply custom CSS 
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stApp {
            background-color: #f7f7f7;
        }
        .title {
            font-size: 42px !important;
            margin-bottom: 10px;
            color: #1E88E5;
            text-align: center;
        }
        .subtitle {
            font-size: 24px !important;
            margin-bottom: 20px;
            color: #2962FF;
            text-align: center;
        }
        .row-title {
            font-size: 28px !important;
            margin: 20px 0px 15px 0px;
            color: #1565C0;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 10px;
        }
        .info-box {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #2196F3;
        }
        .error-box {
            background-color: #ffebee;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #f44336;
        }
        .success-box {
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #4caf50;
        }
        .warning-box {
            background-color: #fff8e1;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #ff9800;
        }
        .stAudio {
            width: 100% !important;
        }
        .section-title {
            color: #1E88E5;
            font-size: 1.5rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        .tips-box {
            background-color: #e8eaf6;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 5px solid #3f51b5;
        }
        .voice-metrics {
            background-color: #e0f7fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 5px solid #00acc1;
        }
        .output-section {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0px;
            border: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display header
    st.markdown("<h1 class='title'>🎙️ Advanced Voice Cloner</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subtitle'>High Quality Voice Cloning with YourTTS</h2>", unsafe_allow_html=True)
    
    # Show installation instructions
    st.info("""
    ### Installation in Progress
    The Voice Cloner application is currently being installed on Streamlit Cloud.
    
    To run this application locally:
    1. Clone the repository: `git clone https://github.com/premsingh23/Voice-Cloner--Live-LLM-using-GPT.git`
    2. Install dependencies: `pip install -r requirements.txt`
    3. Run the app: `streamlit run app.py`
    
    For the best experience, use a machine with GPU support.
    """)
    
    # Show loading status
    try:
        # Try to import the TTS module
        import importlib.util
        tts_spec = importlib.util.find_spec("TTS")
        
        if tts_spec is not None:
            st.success("✅ TTS module found! The application is being prepared.")
        else:
            st.warning("⚠️ TTS module not found. Installation may be in progress.")
            
        # Check other dependencies
        libraries = ["numpy", "pandas", "librosa", "soundfile", "torch"]
        missing = []
        
        for lib in libraries:
            try:
                importlib.import_module(lib)
            except ImportError:
                missing.append(lib)
        
        if missing:
            st.warning(f"⚠️ The following libraries are not yet installed: {', '.join(missing)}")
        else:
            st.success("✅ All required libraries are installed.")
            
    except Exception as e:
        st.error(f"❌ Error checking dependencies: {str(e)}")
    
    # Show GitHub info
    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <h3>Visit the GitHub Repository</h3>
        <a href="https://github.com/premsingh23/Voice-Cloner--Live-LLM-using-GPT" target="_blank">
            Voice Cloner - Advanced TTS with GPT Integration
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px; margin-top: 50px;'>
    Built with ❤️ by Principia Team
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 