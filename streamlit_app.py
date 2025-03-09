import os
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_cloner_app")

# Set environment variable for CPU-only operation on Streamlit Cloud
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Check if running on Streamlit Cloud
is_streamlit_cloud = os.environ.get('STREAMLIT_SHARING', '') == 'true' or os.environ.get('STREAMLIT_CLOUD', '') == 'true'

# Import the main app with error handling
try:
    from app import (
        load_tts_model, 
        analyze_voice_characteristics, 
        process_audio_with_tts_to_file, 
        safe_file_creation,
        convert_media_to_wav,
        download_youtube_video,
        adjust_speaking_rate,
        adjust_pitch,
        smooth_audio,
        combine_audio_samples,
        optimize_text_for_better_speech
    )
    import_success = True
except ImportError as e:
    import_success = False
    import_error = str(e)
    logger.error(f"Import error: {import_error}")

# Main streamlit app
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
    
    # Check if imports were successful
    if not import_success:
        st.error(f"""
        ❌ **Failed to import required modules**
        
        Error details: {import_error}
        
        This app requires specific modules to function correctly. Please check the GitHub repository for installation instructions.
        
        Common issues:
        - TTS package installation problems
        - Missing system dependencies
        - PyTorch compatibility issues
        """)
        
        st.markdown("""
        <div class='info-box'>
            <h3>Troubleshooting steps:</h3>
            <ol>
                <li>Check the app logs for detailed error information</li>
                <li>Verify all system dependencies in packages.txt are installed</li>
                <li>Ensure Python package versions in requirements.txt are compatible</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-top: 30px;'>
            <a href="https://github.com/premsingh23/Voice-Cloner--Live-LLM-using-GPT" target="_blank">
                View on GitHub for installation instructions
            </a>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Streamlit Cloud warning
    if is_streamlit_cloud:
        st.warning("""
        ⚠️ This app is running on Streamlit Cloud without GPU acceleration. Voice generation will be significantly slower.
        For better performance, consider running this app locally with GPU support.
        """)
    
    # Initialize session state
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    
    if 'voice_characteristics' not in st.session_state:
        st.session_state.voice_characteristics = {
            'has_analysis': False,
            'speaking_rate': None,
            'pitch_mean': None,
            'pitch_range': None
        }
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Voice Settings")
        
        # Language selection
        language_options = {
            "English": "en",
            "French": "fr-fr",
            "Portuguese": "pt-br",
            "Spanish": "es",
            "German": "de",
            "Italian": "it"
        }
        language = st.selectbox("Language", list(language_options.keys()), index=0)
        language_code = language_options[language]
        
        # Voice matching options
        match_voice_characteristics = st.checkbox("Match Reference Voice Style", value=True, 
                                   help="Analyze and match the speaking rate and style of the reference voice")
        
        # Speaker reference type selection
        speaker_wav_setting = st.radio(
            "Speaker reference type",
            ["Local File", "Multiple Files", "YouTube Link"],
            index=0
        )
        
        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            smoothing_level = st.slider("Smoothing Level", min_value=0, max_value=21, value=0, 
                               help="Set smoothing level. 0 disables smoothing")
    
    # Load TTS model
    try:
        with st.spinner("Loading YourTTS model..."):
            model, device = load_tts_model()
        st.success("✅ YourTTS model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
    
    # FIRST ROW - Upload and Text Prompt
    st.markdown("<div class='row-title'>Voice Cloning Process</div>", unsafe_allow_html=True)
    upload_col, text_col = st.columns(2)
    
    # Upload column
    with upload_col:
        st.markdown("<div class='section-title'>1. Upload a Voice Sample</div>", unsafe_allow_html=True)
        
        speaker_wav = None
        if speaker_wav_setting == "Local File":
            uploaded_media = st.file_uploader("Upload a media file (Audio or Video)", type=["wav", "mp3", "ogg", "m4a", "mp4", "mkv", "avi"])
            if uploaded_media is not None:
                try:
                    speaker_wav = safe_file_creation(uploaded_media, f".{uploaded_media.name.split('.')[-1]}")
                    if not speaker_wav.lower().endswith('.wav'):
                        speaker_wav = convert_media_to_wav(speaker_wav)
                    st.audio(speaker_wav, format="audio/wav")
                    
                    # Analyze voice characteristics if matching is enabled
                    if match_voice_characteristics:
                        st.session_state.voice_characteristics = analyze_voice_characteristics(speaker_wav)
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")
                    speaker_wav = None
                    
        elif speaker_wav_setting == "Multiple Files":
            uploaded_files = st.file_uploader(
                "Upload audio files (WAV, MP3, etc.)",
                type=["wav", "mp3", "m4a", "ogg", "flac", "mp4", "avi", "mov", "mkv"],
                accept_multiple_files=True,
                help="Upload multiple clear audio samples of the voice you want to clone",
            )
            speaker_wav = None
            if uploaded_files and len(uploaded_files) > 0:
                try:
                    with st.spinner(f"Processing {len(uploaded_files)} audio samples..."):
                        speaker_wav = combine_audio_samples(uploaded_files)
                        if speaker_wav:
                            st.audio(speaker_wav)
                            st.success(f"Successfully combined {len(uploaded_files)} audio samples for voice cloning")
                            
                            # Analyze voice characteristics if matching is enabled
                            if match_voice_characteristics:
                                st.session_state.voice_characteristics = analyze_voice_characteristics(speaker_wav)
                except Exception as e:
                    st.error(f"Error processing uploaded files: {str(e)}")
                    
        elif speaker_wav_setting == "YouTube Link":
            youtube_url = st.text_input(
                "YouTube URL",
                help="Enter a YouTube link with the voice you want to clone",
            )
            speaker_wav = None
            if youtube_url and youtube_url.strip():
                try:
                    with st.spinner("Downloading audio from YouTube..."):
                        speaker_wav = download_youtube_video(youtube_url)
                        if speaker_wav:
                            st.audio(speaker_wav)
                            st.success("Successfully downloaded audio from YouTube")
                            
                            # Analyze voice characteristics if matching is enabled
                            if match_voice_characteristics:
                                st.session_state.voice_characteristics = analyze_voice_characteristics(speaker_wav)
                except Exception as e:
                    st.error(f"Error downloading from YouTube: {str(e)}")
    
    # Text input column
    with text_col:
        st.markdown("<div class='section-title'>2. Enter Text to Synthesize</div>", unsafe_allow_html=True)
        text_to_speak = st.text_area("Text to convert to speech", 
                         "In a high-tech digital lab, a mere second voice clip sparked an incredible journey. The mode listened intently to every syllable and inflection, gradually absorbing the unique tone and rhythm of the speaker. Each brief sound became a lesson in character, transforming that tiny snippet into a vivid, lifelike digital echo. Even with so little data, the system learned to capture the soul of the voice, proving that sometimes, less truly is more.", 
                         height=150)
    
    # SECOND ROW - Generate Button (centered)
    st.markdown("<div style='text-align: center; margin: 20px 0px;'>", unsafe_allow_html=True)
    generate_button = st.button("Generate Voice", 
                              disabled=(speaker_wav_setting == "Local File" and speaker_wav is None) or 
                                       (speaker_wav_setting == "Multiple Files" and speaker_wav is None) or 
                                       (speaker_wav_setting == "YouTube Link" and (youtube_url is None or youtube_url.strip() == "")), 
                              type="primary", 
                              use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Voice generation
    if generate_button:
        if not text_to_speak.strip():
            st.error("Please enter some text to synthesize.")
        else:
            with st.spinner(f"Generating voice with {language.upper()} settings..."):
                try:
                    # Generate the cloned voice
                    import time
                    start_time = time.time()
                    
                    # Pre-process the text for better results
                    voice_chars = st.session_state.voice_characteristics if match_voice_characteristics else None
                    text_to_process = optimize_text_for_better_speech(text_to_speak, language_code, voice_chars)
                    
                    logger.info(f"Generating voice with text: {text_to_process[:50]}...")
                    
                    if speaker_wav:
                        # Generate the voice to a file for better quality
                        output_file = process_audio_with_tts_to_file(
                            model=model,
                            text=text_to_process,
                            speaker_wav=speaker_wav,
                            language=language_code
                        )
                    else:
                        # Use default speaker
                        wav = model.tts(
                            text=text_to_process,
                            language=language_code
                        )
                        # Save the generated audio
                        import tempfile
                        import soundfile as sf
                        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                        sf.write(output_file, wav, 22050)  # Always use optimal sample rate
                    
                    # Add to session state for cleanup
                    st.session_state.temp_files.append(output_file)
                    
                    generation_time = time.time() - start_time
                    logger.info(f"Voice generation completed in {generation_time:.2f} seconds")
                    
                    # Adjust speaking rate to match reference voice if matching is enabled
                    if match_voice_characteristics and st.session_state.voice_characteristics.get('speaking_rate', 0) > 0:
                        target_rate = st.session_state.voice_characteristics['speaking_rate']
                        output_file = adjust_speaking_rate(output_file, target_rate)

                    # Adjust pitch to match reference voice if matching is enabled
                    if match_voice_characteristics and st.session_state.voice_characteristics.get('pitch_mean', 0) > 0:
                        target_pitch = st.session_state.voice_characteristics['pitch_mean']
                        output_file = adjust_pitch(output_file, target_pitch)

                    # Apply smoothing to further refine voice smoothness to match the sample voice
                    if match_voice_characteristics and smoothing_level > 0:
                        output_file = smooth_audio(output_file, window_length=smoothing_level)

                    # Display success message and audio
                    st.markdown("<div class='output-section'>", unsafe_allow_html=True)
                    st.markdown("<div class='section-title'>3. Generated Voice</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='success-box'>✅ Voice generated successfully in {generation_time:.2f} seconds!</div>", unsafe_allow_html=True)
                    
                    # Play the generated audio
                    st.audio(output_file, format="audio/wav")
                    
                    # If voice matching was used, show which characteristics were matched
                    if match_voice_characteristics and st.session_state.voice_characteristics['has_analysis']:
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.markdown("**Voice Style Matching Applied:**")
                        st.markdown("• Speaking rate matched to reference audio")
                        st.markdown("• Speech pacing and rhythm adapted")
                        st.markdown("• Text optimized for the detected speaking style")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add download button
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"voice_clone_{timestamp}.wav"
                    
                    with open(output_file, "rb") as file:
                        st.download_button(
                            label="Download Generated Voice",
                            data=file,
                            file_name=download_filename,
                            mime="audio/wav"
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    error_msg = f"Error generating voice: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
    
    # THIRD ROW - Tips Section
    st.markdown("<div class='row-title' style='margin-top: 40px;'>Tips for Better Results</div>", unsafe_allow_html=True)
    
    # Voice Tips column
    voice_tips_col, text_tips_col = st.columns(2)
    
    # Voice Tips column
    with voice_tips_col:
        with st.expander("4. Voice Sample Tips", expanded=True):
            st.markdown("""
            - **Recording environment:** Use a quiet room with minimal echo and background noise
            - **Microphone:** Use a good quality microphone, avoid laptop/phone built-in mics if possible
            - **Speaking style:** Speak clearly with consistent volume and tone
            - **Duration:** Longer samples (30+ seconds) generally give better results
            - **Content:** Varied speech with different sentence types works best
            - **Processing:** Advanced processing is automatically applied for optimal results
            """)
    
    # Text Tips column
    with text_tips_col:
        with st.expander("5. Text Content Tips", expanded=True):
            st.markdown("""
            - **Sentence length:** Medium-length sentences work better than very short or very long ones
            - **Punctuation:** Ensure proper punctuation for natural phrasing
            - **Language match:** The language selected should match the text language
            - **Phonetics:** Consider phonetic spelling for proper pronunciation of names
            - **Simplicity:** Start with simple sentences before trying complex ones
            """)
    
    # Footer with GitHub link
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
    Built with ❤️ by Principia Team | 
    <a href="https://github.com/premsingh23/Voice-Cloner--Live-LLM-using-GPT" target="_blank">GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Display app info
    st.markdown("### About")
    st.markdown("""
    This application uses YourTTS, a zero-shot multi-speaker TTS model, to clone voices from audio samples.
    
    **Features:**
    - Upload any voice recording to clone
    - Supports multiple languages
    - Advanced audio preprocessing for optimal results
    - Voice style matching to replicate speaking rate and pattern
    - Download generated voices for later use
    
    **Technical Details:**
    - Model: YourTTS (Coqui TTS)
    - Audio processing: High-pass filtering, silence trimming, normalization
    - Voice analysis: Speaking rate detection, pitch analysis
    - Optimal sample rate: 22050 Hz
    """)

if __name__ == "__main__":
    main() 