import os
import time
import tempfile
import streamlit as st
import torch
from TTS.api import TTS
from utils import convert_audio_for_model, save_audio, cleanup_temp_files
from utils import check_ffmpeg, check_cuda_availability, validate_file_exists
from check_cuda import check_cuda
import logging
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
import shutil
import re
import subprocess
from pytube import YouTube
import yt_dlp
import warnings
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_cloner_app")

# Page configuration
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
    .row-title {
        font-size: 28px !important;
        margin: 20px 0px 15px 0px;
        color: #1565C0;
        text-align: center;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state to store temporary file paths for cleanup
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

# Initialize session state for error tracking
if 'error_logs' not in st.session_state:
    st.session_state.error_logs = []

# Initialize session state for reference voice characteristics
if 'voice_characteristics' not in st.session_state:
    st.session_state.voice_characteristics = {
        'speaking_rate': 0,
        'pitch_mean': 0,
        'pitch_std': 0,
        'energy_mean': 0,
        'has_analysis': False
    }

def log_error(error_message):
    """Log an error message and store it in session state"""
    logger.error(error_message)
    st.session_state.error_logs.append(f"{time.strftime('%H:%M:%S')} - {error_message}")

@st.cache_resource
def load_tts_model():
    """
    Load the TTS model with caching to avoid reloading.
    
    Returns:
        TTS: The loaded TTS model
    """
    # Check if CUDA is available
    device, device_info = check_cuda_availability()
    cuda_available = (device == "cuda")
    
    try:
        # Load YourTTS model
        logger.info("Loading YourTTS model...")
        model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", 
                    progress_bar=True,
                    gpu=cuda_available)
        logger.info("YourTTS model loaded successfully")
        return model, device
    except Exception as e:
        error_msg = f"Error loading TTS model: {str(e)}"
        log_error(error_msg)
        st.error(error_msg)
        raise

def cleanup_session_files():
    """Clean up temporary files created in this session"""
    if 'temp_files' in st.session_state:
        logger.info(f"Cleaning up {len(st.session_state.temp_files)} temporary files")
        for file_path in st.session_state.temp_files:
            cleanup_temp_files(file_path)
        st.session_state.temp_files = []

def safe_file_creation(file_obj, extension):
    """
    Safely create a temporary file from a file object uploaded by the user.
    
    Args:
        file_obj: The file object from st.file_uploader
        extension: File extension including the dot (e.g., '.wav')
        
    Returns:
        str: Path to the created temporary file, or None if an error occurred
    """
    try:
        # Create a temporary file with a unique name
        temp_dir = tempfile.gettempdir()
        unique_filename = f"upload_{int(time.time())}_{os.path.basename(file_obj.name)}"
        temp_path = os.path.join(temp_dir, unique_filename)
        
        logger.info(f"Creating temporary file: {temp_path}")
        
        # Write the file
        with open(temp_path, 'wb') as f:
            f.write(file_obj.getvalue())
        
        # Verify the file was created
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Failed to create temporary file: {temp_path}")
            
        # Check if the file has content
        if os.path.getsize(temp_path) == 0:
            raise ValueError(f"Created file is empty: {temp_path}")
        
        logger.info(f"Successfully created temporary file: {temp_path}")
        return temp_path
    except Exception as e:
        error_msg = f"Error creating temporary file: {str(e)}"
        log_error(error_msg)
        return None

def analyze_voice_characteristics(audio_file):
    """
    Analyze the voice characteristics of the reference audio
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        dict: Dictionary of voice characteristics
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Calculate duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract speech parts only (remove silence)
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        y_speech = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        
        # Calculate pitch (F0) using PYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(y_speech, 
                                                   fmin=librosa.note_to_hz('C2'), 
                                                   fmax=librosa.note_to_hz('C7'),
                                                   sr=sr)
        
        # Filter out NaN values (unvoiced segments)
        f0 = f0[~np.isnan(f0)]
        
        # Calculate pitch statistics
        if len(f0) > 0:
            pitch_mean = np.mean(f0)
            pitch_std = np.std(f0)
        else:
            pitch_mean = 0
            pitch_std = 0
        
        # Calculate energy/amplitude envelope
        energy = np.abs(y_speech)
        energy_mean = np.mean(energy)
        
        # Calculate speaking rate (approximate syllables per second)
        # Use amplitude envelope to find peaks (syllable nuclei)
        energy_envelope = librosa.onset.onset_strength(y=y_speech, sr=sr)
        peaks = librosa.util.peak_pick(energy_envelope, 3, 3, 3, 5, 0.5, 10)
        if duration > 0:
            speaking_rate = len(peaks) / duration
        else:
            speaking_rate = 0
        
        voice_metrics = {
            'duration': duration,
            'speaking_rate': speaking_rate,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'energy_mean': energy_mean,
            'has_analysis': True
        }
        
        logger.info(f"Voice analysis complete: {voice_metrics}")
        return voice_metrics
        
    except Exception as e:
        logger.error(f"Error analyzing voice: {str(e)}")
        return {
            'speaking_rate': 0,
            'pitch_mean': 0,
            'pitch_std': 0,
            'energy_mean': 0,
            'has_analysis': False
        }

def optimize_audio_for_cloning(audio_file):
    """
    Optimize audio for the best possible voice cloning results.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        str: Path to the optimized audio file
    """
    try:
        # Load audio using librosa for advanced processing
        y, sr = librosa.load(audio_file, sr=None)
        
        # Trim silence aggressively (top_db=30 is more aggressive than the default 60)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        
        # If the trimmed audio is too short (less than 1 second), use original
        if len(y_trimmed) / sr < 1.0:
            y_trimmed = y
        
        # Normalize audio for consistent volume
        y_normalized = librosa.util.normalize(y_trimmed)

        # Apply noise reduction if available
        try:
            import noisereduce as nr
            y_denoised = nr.reduce_noise(y=y_normalized, sr=sr)
            logger.info("Noise reduction applied.")
        except ImportError:
            logger.warning("noisereduce package not installed; skipping noise reduction.")
            y_denoised = y_normalized
        
        # Apply a slight high-pass filter to reduce low-frequency noise (common in recordings)
        # This helps YourTTS focus on the voice characteristics
        b, a = librosa.filters.butter(4, 100/(sr/2), btype='highpass')
        y_filtered = librosa.filtfilt(b, a, y_denoised)
        
        # Resample to 22050 Hz (YourTTS optimal sample rate)
        if sr != 22050:
            y_resampled = librosa.resample(y_filtered, orig_sr=sr, target_sr=22050)
        else:
            y_resampled = y_filtered
        
        # Create output file
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        
        # Save the processed audio
        sf.write(output_file, y_resampled, 22050)
        
        # Analyze voice characteristics
        st.session_state.voice_characteristics = analyze_voice_characteristics(output_file)
        
        logger.info(f"Audio optimized for cloning: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error optimizing audio: {str(e)}")
        # Fallback to basic conversion if advanced processing fails
        return convert_audio_for_model(audio_file, model_sample_rate=22050, trim_silence=True, normalize=True)

def process_audio_with_tts_to_file(model, text, speaker_wav, language, output_file=None):
    """
    Generate cloned voice using the tts_to_file method for improved quality
    
    Args:
        model: TTS model
        text: Text to convert to speech
        speaker_wav: Path to the speaker reference audio file
        language: Language code
        output_file: Path to save the output file, if None a temp file will be created
        
    Returns:
        Path to the generated audio file
    """
    if output_file is None:
        # Create a temporary output file
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    
    try:
        logger.info(f"Generating voice to file: {output_file}")
        logger.info(f"Using reference audio: {speaker_wav}")
        logger.info(f"Text: '{text}'")
        logger.info(f"Language: {language}")
        
        # Verify the speaker_wav exists
        if not os.path.exists(speaker_wav):
            raise ValueError(f"Speaker wav file not found: {speaker_wav}")
        
        # Generate the voice using the better tts_to_file method
        model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_file
        )
        
        # Verify the output file was created
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file was not created: {output_file}")
        
        return output_file
    except Exception as e:
        # Clean up the output file if it was created but there was an error
        if os.path.exists(output_file):
            os.unlink(output_file)
        raise e

def optimize_text_for_better_speech(text, language_code, voice_characteristics=None):
    """
    Optimize text for better speech synthesis with YourTTS.
    
    Args:
        text: Original text
        language_code: Language code
        voice_characteristics: Voice characteristics to match
        
    Returns:
        Optimized text
    """
    # Add appropriate punctuation and formatting to improve voice quality
    
    # Make sure text ends with punctuation
    if not text or len(text) == 0:
        return text
        
    if text[-1] not in ['.', '!', '?', ',', ';', ':', '"', "'"]:
        text = text + '.'
    
    # Adjust punctuation based on speaking rate if available
    if voice_characteristics and voice_characteristics['has_analysis']:
        speaking_rate = voice_characteristics['speaking_rate']
        
        # Add more or less pauses based on speaking rate
        if speaking_rate > 4.0:  # Very fast speaker
            # Reduce pauses by converting some periods to commas
            text = re.sub(r'(?<!\.)\.(?!\.)', ',', text)
        elif speaking_rate < 2.5:  # Slow speaker
            # Add more pauses with commas and ellipses
            text = text.replace('. ', '... ')
            text = re.sub(r'(\w+)(\s+)(\w+)', r'\1, \3', text)
    
    # Break very long sentences
    if len(text) > 150 and '.' not in text[:-1]:
        words = text.split()
        if len(words) > 20:
            midpoint = len(words) // 2
            # Find a good place to break near the midpoint
            for i in range(midpoint-3, midpoint+3):
                if i > 0 and i < len(words):
                    words[i] = words[i] + ','
                    break
            text = ' '.join(words)
    
    # Language-specific optimizations
    if language_code.startswith('en'):
        # For English, add spacing after punctuation if missing
        text = text.replace('.','. ').replace('!','! ').replace('?','? ')
        text = text.replace('  ', ' ')  # Fix double spaces
    
    # Remove multiple spaces
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    return text

# Insert new function to adjust speaking rate based on reference sample

def adjust_speaking_rate(audio_file, target_speaking_rate, tolerance=0.1):
    """Adjust the speaking rate of the generated audio to match the target rate.

    Args:
        audio_file (str): Path to the generated audio file.
        target_speaking_rate (float): Desired syllables per second.
        tolerance (float): Tolerance ratio (default 0.1) for no adjustment.

    Returns:
        str: Path to the adjusted audio file if adjustment was made, otherwise the original file.
    """
    try:
        y, sr = librosa.load(audio_file, sr=None)
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        if len(non_silent_intervals) > 0:
            y_speech = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        else:
            y_speech = y
        onset_env = librosa.onset.onset_strength(y=y_speech, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        duration = librosa.get_duration(y=y, sr=sr)
        current_rate = len(peaks) / duration if duration > 0 else 0
        if current_rate == 0 or abs(current_rate - target_speaking_rate) / target_speaking_rate < tolerance:
            logger.info("Speaking rate within tolerance; no adjustment needed.")
            return audio_file
        # Calculate time stretch factor
        factor = target_speaking_rate / current_rate
        y_stretched = librosa.effects.time_stretch(y, factor)
        new_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        sf.write(new_output_file, y_stretched, sr)
        logger.info(f"Time-stretch adjustment applied with factor {factor:.2f}.")
        return new_output_file
    except Exception as e:
        logger.error(f"Error adjusting speaking rate: {str(e)}")
        return audio_file

def adjust_pitch(audio_file, target_pitch, tolerance=2.0):
    """Adjust the pitch of the generated audio to match the target pitch.
    
    Args:
        audio_file (str): Path to the audio file.
        target_pitch (float): Desired average pitch in Hz.
        tolerance (float): Tolerance in Hz for adjustment (default 2.0).
        
    Returns:
        str: Path to the adjusted audio file if adjustment was made, otherwise the original file.
    """
    try:
        y, sr = librosa.load(audio_file, sr=None)
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        if len(non_silent_intervals) > 0:
            y_speech = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        else:
            y_speech = y
        f0, _, _ = librosa.pyin(y_speech, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) > 0:
            current_pitch = np.mean(f0)
        else:
            current_pitch = 0
        if current_pitch == 0 or abs(current_pitch - target_pitch) < tolerance:
            logger.info("Pitch within tolerance; no adjustment needed.")
            return audio_file
        shift_semitones = 12 * np.log2(target_pitch / current_pitch)
        y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=shift_semitones)
        new_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        sf.write(new_output_file, y_shifted, sr)
        logger.info(f"Pitch adjustment applied with shift {shift_semitones:.2f} semitones.")
        return new_output_file
    except Exception as e:
        logger.error(f"Error adjusting pitch: {str(e)}")
        return audio_file

def smooth_audio(audio_file, window_length=21):
    """Smooth the generated audio using a moving average filter to match the sample's natural smoothness.
    
    Args:
        audio_file (str): Path to the audio file.
        window_length (int): Length of the smoothing window (default 21).
        
    Returns:
        str: Path to the smoothed audio file.
    """
    try:
        y, sr = librosa.load(audio_file, sr=None)
        if len(y) < window_length:
            window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
        window = np.ones(window_length) / window_length
        y_smooth = np.convolve(y, window, mode='same')
        new_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        sf.write(new_output_file, y_smooth, sr)
        logger.info(f"Audio smoothing applied with window length {window_length}.")
        return new_output_file
    except Exception as e:
        logger.error(f"Error smoothing audio: {str(e)}")
        return audio_file

def iterative_voice_cloning(model, text, speaker_wav, language, iterations=5):
    """Iteratively refine the generated voice to more closely match the reference sample.

    Args:
        model: TTS model.
        text: Text to synthesize.
        speaker_wav: Path to the reference speaker sample audio.
        language: Language code.
        iterations (int, default 5): Number of iterative refinements.
        
    Returns:
        str: Path to the final generated audio file.
    """
    current_reference = speaker_wav
    for i in range(iterations):
        logger.info(f"Iterative cloning pass {i+1}/{iterations} using reference: {current_reference}")
        output_file = process_audio_with_tts_to_file(model=model, text=text, speaker_wav=current_reference, language=language)
        current_reference = output_file
    return current_reference

def convert_media_to_wav(input_file, sample_rate=22050):
    """Convert any media file (audio/video) to WAV using ffmpeg."""
    try:
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        cmd = f'ffmpeg -y -i "{input_file}" -ar {sample_rate} -ac 1 -f wav "{output_file}"'
        logger.info(f"Running ffmpeg command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        return output_file
    except Exception as e:
        logger.error(f"Error converting media to wav: {str(e)}")
        return None

def download_youtube_video(youtube_url):
    """Download video from YouTube and return the local file path.
    Tries yt-dlp first, then falls back to pytube if that fails.
    """
    logger.info(f"Attempting to download YouTube video: {youtube_url}")
    output_path = None
    
    # First try with yt-dlp (more resilient against YouTube restrictions)
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f"{tempfile.gettempdir()}/%(title)s-%(id)s.%(ext)s",
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            output_path = ydl.prepare_filename(info)
            logger.info(f"YouTube video downloaded with yt-dlp: {output_path}")
            return output_path
    except Exception as e:
        logger.warning(f"yt-dlp download failed: {str(e)}, falling back to pytube")
    
    # Fallback to pytube if yt-dlp fails
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(only_audio=True).first() or yt.streams.get_highest_resolution()
        output_path = stream.download()
        logger.info(f"YouTube video downloaded with pytube: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Both yt-dlp and pytube failed. Error: {str(e)}")
        return None

def transfer_prosody(generated_file, reference_file, frame_length=2048, hop_length=512, pitch_threshold=5.0, strength=0.5):
    """Transfer local prosody (accent/emphasis) from the reference audio to the generated audio.
    This function computes the pitch contours of both audios, calculates the local correction in semitones,
    and applies a local pitch shift to each frame if the difference exceeds a threshold.
    
    Args:
        generated_file: Path to the generated audio file
        reference_file: Path to the reference audio file
        frame_length: Frame length for pitch analysis
        hop_length: Hop length for pitch analysis
        pitch_threshold: Threshold for applying pitch correction
        strength: Strength of the pitch correction (0.0 to 1.0)
    """
    try:
        y_gen, sr = librosa.load(generated_file, sr=None)
        y_ref, _ = librosa.load(reference_file, sr=sr)

        # Extract pitch contours
        f0_gen, _, _ = librosa.pyin(y_gen, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=frame_length, hop_length=hop_length)
        f0_ref, _, _ = librosa.pyin(y_ref, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=frame_length, hop_length=hop_length)

        # Replace NaNs with zeros
        f0_gen = np.nan_to_num(f0_gen, nan=0.0)
        f0_ref = np.nan_to_num(f0_ref, nan=0.0)

        # Handle different lengths by interpolating reference pitch to match generated pitch length
        if len(f0_gen) != len(f0_ref):
            logger.info(f"Pitch contour length mismatch: generated={len(f0_gen)}, reference={len(f0_ref)}. Interpolating...")
            # Create a time axis for both contours
            x_ref = np.linspace(0, 1, len(f0_ref))
            x_gen = np.linspace(0, 1, len(f0_gen))
            
            # Interpolate the reference pitch contour to match the generated pitch contour length
            from scipy.interpolate import interp1d
            f_interp = interp1d(x_ref, f0_ref, kind='linear', bounds_error=False, fill_value=0)
            f0_ref_resampled = f_interp(x_gen)
            f0_ref = f0_ref_resampled
            
            logger.info(f"After interpolation: generated={len(f0_gen)}, reference={len(f0_ref)}")

        # Compute correction contour in semitones
        eps = 1e-6
        correction = 12 * np.log2((f0_ref + eps) / (f0_gen + eps))
        # Smooth the correction contour
        correction_smooth = np.convolve(correction, np.ones(5)/5, mode='same')

        # Instead of processing frame by frame, process the entire audio at once with a global pitch shift
        # This is more robust and avoids resampling errors with small segments
        try:
            # Calculate the average pitch correction, ignoring extreme values
            valid_corrections = correction_smooth[np.abs(correction_smooth) < 12]  # Ignore octave+ shifts
            if len(valid_corrections) > 0:
                avg_correction = np.mean(valid_corrections)
                # Apply strength factor to scale the correction (0.5 = half the correction)
                scaled_correction = avg_correction * strength
                logger.info(f"Original correction: {avg_correction:.2f} semitones, scaled to {scaled_correction:.2f} with strength {strength}")
                
                # Only apply if the correction is significant
                if abs(avg_correction) > 0.5:  # At least half a semitone difference
                    y_out = librosa.effects.pitch_shift(y=y_gen, sr=sr, n_steps=scaled_correction)
                else:
                    logger.info("Pitch correction too small, skipping prosody transfer")
                    y_out = y_gen
            else:
                logger.info("No valid pitch corrections found, skipping prosody transfer")
                y_out = y_gen
        except Exception as e:
            logger.error(f"Error during global pitch shift: {str(e)}")
            y_out = y_gen  # Fallback to original audio
        
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        sf.write(output_file, y_out, sr)
        logger.info("Prosody transfer applied.")
        return output_file
    except Exception as e:
        logger.error(f"Error in prosody transfer: {str(e)}")
        return generated_file

def combine_audio_samples(file_list):
    """
    Process multiple audio files and combine their characteristics for better voice cloning.
    
    Args:
        file_list: List of audio file paths or uploaded file objects
        
    Returns:
        path to a combined audio file that contains relevant segments from all samples
    """
    if not file_list or len(file_list) == 0:
        logger.error("No audio files provided to combine")
        return None
        
    # If only one file, just process it normally
    if len(file_list) == 1:
        if isinstance(file_list[0], str):  # It's already a file path
            return file_list[0]
        else:  # It's an uploaded file object
            return safe_file_creation(file_list[0], extension="wav")
    
    logger.info(f"Combining {len(file_list)} audio files for voice cloning")
    combined_segments = []
    sample_rate = None
    
    # Process each file
    for i, file_obj in enumerate(file_list):
        try:
            # Convert uploaded file to path if needed
            if not isinstance(file_obj, str):
                temp_path = safe_file_creation(file_obj, extension="wav")
            else:
                temp_path = file_obj
                
            # Convert to WAV if it's not already
            if not temp_path.lower().endswith('.wav'):
                temp_path = convert_media_to_wav(temp_path)
                
            # Load audio
            y, sr = librosa.load(temp_path, sr=None)
            
            # Set sample rate from first file
            if sample_rate is None:
                sample_rate = sr
            
            # Resample if necessary
            if sr != sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
            
            # Extract 3-5 seconds of clear speech from each file
            if len(y) > 3 * sample_rate:
                # Use VAD to find speech segments, or just take the middle 3-5 seconds
                segment_length = min(5 * sample_rate, len(y) // 2)
                start_idx = len(y) // 4  # Skip the first quarter of the file
                combined_segments.append(y[start_idx:start_idx + segment_length])
            else:
                combined_segments.append(y)
                
            logger.info(f"Processed sample {i+1}/{len(file_list)}")
            
        except Exception as e:
            logger.error(f"Error processing audio file {i+1}: {str(e)}")
    
    # Combine segments with short silences between them
    silence = np.zeros(int(0.5 * sample_rate))  # 0.5 second silence
    combined_audio = np.concatenate([seg for pair in zip(combined_segments, [silence] * len(combined_segments)) for seg in pair][:-1])
    
    # Save combined audio
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    sf.write(output_file, combined_audio, sample_rate)
    
    logger.info(f"Created combined audio file from {len(file_list)} samples: {output_file}")
    return output_file

def main():
    """Main function to run the Streamlit application."""
    
    # Register cleanup for session end
    try:
        import atexit
        atexit.register(cleanup_session_files)
    except Exception as e:
        logger.warning(f"Failed to register cleanup function: {str(e)}")
    
    # Display header
    st.markdown("<h1 class='title'>🎙️ Advanced Voice Cloner</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subtitle'>High Quality Voice Cloning with YourTTS</h2>", unsafe_allow_html=True)
    
    # Display GPU status
    device, device_info = check_cuda_availability()
    if device == "cuda":
        gpu_info = f"CUDA is available: {device_info['name']} with CUDA {device_info['cuda_version']}"
        st.markdown(f"<div class='success-box'>{gpu_info}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='error-box'>⚠️ CUDA is not available. Voice generation will be slow on CPU.</div>", unsafe_allow_html=True)
    
    # Check FFmpeg availability
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        st.warning("⚠️ FFmpeg not found. Audio conversion may not work correctly. Please install FFmpeg and add it to your PATH.")
    
    # Create sidebar
    st.sidebar.title("Settings")
    
    # Debug toggle in sidebar (collapsed by default)
    with st.sidebar.expander("Debug Options", expanded=False):
        show_logs = st.checkbox("Show Error Logs", value=False)
        
        if show_logs and len(st.session_state.error_logs) > 0:
            st.markdown("### Error Logs")
            for log in st.session_state.error_logs:
                st.text(log)
            
            if st.button("Clear Logs"):
                st.session_state.error_logs = []
                st.experimental_rerun()
    
    # Model loading status
    with st.sidebar:
        st.markdown("### Model Status")
        model_load_state = st.info("Loading YourTTS model...")
    
    # Load the TTS model
    try:
        model, device = load_tts_model()
        with st.sidebar:
            model_load_state.success("✅ YourTTS model loaded successfully")
    except Exception as e:
        with st.sidebar:
            model_load_state.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
    
    # Add minimal settings to sidebar
    with st.sidebar:
        st.markdown("### Voice Settings")
        
        # Language selection with more specific options
        language_options = {
            "English": "en",
            "French": "fr-fr",  # More specific French code
            "Portuguese": "pt-br",  # Brazilian Portuguese
            "Spanish": "es",
            "German": "de",
            "Italian": "it"
        }
        language = st.selectbox("Language", list(language_options.keys()), index=0)
        language_code = language_options[language]
        
        # Voice matching options
        match_voice_characteristics = st.checkbox("Match Reference Voice Style", value=True, 
                                   help="Analyze and match the speaking rate and style of the reference voice")
        
        # Speaker reference type selection - moved out of advanced options
        speaker_wav_setting = st.radio(
            "Speaker reference type",
            ["Local File", "Multiple Files", "YouTube Link"],
            index=0
        )
        
        # Advanced cloning toggle
        use_advanced_cloning = st.checkbox("Use Advanced Cloning", value=True, 
                                   help="Applies specialized audio processing for optimal voice cloning")
        
        # Only offer default speaker option collapsed in expander
        with st.expander("Advanced Options", expanded=False):
            smoothing_level = st.slider("Smoothing Level", min_value=0, max_value=21, value=0, help="Set smoothing level. 0 disables smoothing")
    
    # FIRST ROW - Upload and Text Prompt
    st.markdown("<div class='row-title'>Voice Cloning Process</div>", unsafe_allow_html=True)
    upload_col, text_col = st.columns(2)

    # Upload column
    with upload_col:
        st.markdown("<div class='section-title'>1. Upload a Voice Sample</div>", unsafe_allow_html=True)

        if speaker_wav_setting == "Local File":
            uploaded_media = st.file_uploader("Upload a media file (Audio or Video)", type=["wav", "mp3", "ogg", "m4a", "mp4", "mkv", "avi"])
            if uploaded_media is None:
                speaker_wav = None
            else:
                logger.info(f"User uploaded file: {uploaded_media.name}")
                temp_file_path = safe_file_creation(uploaded_media, f".{uploaded_media.name.split('.')[-1]}")
                if temp_file_path:
                    st.session_state.temp_files.append(temp_file_path)
                    ext = uploaded_media.name.split('.')[-1].lower()
                    if ext != "wav":
                        converted_audio = convert_media_to_wav(temp_file_path)
                        speaker_wav = converted_audio
                    else:
                        speaker_wav = temp_file_path
                    if speaker_wav:
                        st.audio(speaker_wav, format="audio/wav")
                else:
                    st.error("Failed to process the uploaded file. Please try again.")
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
                        # Display combined audio for preview
                        if speaker_wav:
                            st.audio(speaker_wav)
                            st.success(f"Successfully combined {len(uploaded_files)} audio samples for voice cloning")
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
                except Exception as e:
                    st.error(f"Error downloading from YouTube: {str(e)}")

    # Text input column
    with text_col:
        st.markdown("<div class='section-title'>2. Enter Text to Synthesize</div>", unsafe_allow_html=True)
        text_to_speak = st.text_area("Text to convert to speech", "In a high-tech digital lab, a mere second voice clip sparked an incredible journey. The mode listened intently to every syllable and inflection, gradually absorbing the unique tone and rhythm of the speaker. Each brief sound became a lesson in character, transforming that tiny snippet into a vivid, lifelike digital echo. Even with so little data, the system learned to capture the soul of the voice, proving that sometimes, less truly is more.", height=150)

    # SECOND ROW - Generate Button (centered)
    st.markdown("<div style='text-align: center; margin: 20px 0px;'>", unsafe_allow_html=True)
    generate_button = st.button("Generate Voice", disabled=(speaker_wav_setting == "Local File" and speaker_wav is None) or 
                                                          (speaker_wav_setting == "Multiple Files" and speaker_wav is None) or 
                                                          (speaker_wav_setting == "YouTube Link" and (youtube_url is None or youtube_url.strip() == "")), 
                                                          type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Output section for generated voice
    if generate_button:
        if not text_to_speak.strip():
            st.error("Please enter some text to synthesize.")
        else:
            with st.spinner(f"Generating voice with {language.upper()} settings..."):
                try:
                    # Generate the cloned voice
                    start_time = time.time()
                    
                    # Pre-process the text for better results
                    voice_chars = st.session_state.voice_characteristics if match_voice_characteristics else None
                    text_to_process = optimize_text_for_better_speech(text_to_speak, language_code, voice_chars)
                    
                    logger.info(f"Generating voice with text: {text_to_process[:50]}...")
                    
                    # Use the improved tts_to_file method for better quality
                    if speaker_wav_setting == "Local File":
                        # Check if speaker reference audio is provided
                        if not speaker_wav:
                            st.error("Speaker reference audio path is empty or None. Please upload a valid audio or video file.")
                            st.stop()
                        validate_file_exists(speaker_wav, "Speaker reference audio")
                        
                        # Generate the voice to a file for better quality
                        output_file = process_audio_with_tts_to_file(
                            model=model,
                            text=text_to_process,
                            speaker_wav=speaker_wav,
                            language=language_code
                        )
                    elif speaker_wav_setting == "Multiple Files":
                        # Check if speaker reference audio is provided
                        if not speaker_wav:
                            st.error("No valid audio samples were processed. Please upload valid audio files.")
                            st.stop()
                        validate_file_exists(speaker_wav, "Combined speaker reference audio")
                        
                        # Generate the voice to a file for better quality
                        output_file = process_audio_with_tts_to_file(
                            model=model,
                            text=text_to_process,
                            speaker_wav=speaker_wav,
                            language=language_code
                        )
                    elif speaker_wav_setting == "YouTube Link":
                        # Check if speaker reference audio is provided
                        if not speaker_wav:
                            st.error("YouTube audio download failed. Please try another YouTube URL.")
                            st.stop()
                        validate_file_exists(speaker_wav, "YouTube reference audio")
                        
                        # Generate the voice to a file for better quality
                        output_file = process_audio_with_tts_to_file(
                            model=model,
                            text=text_to_process,
                            speaker_wav=speaker_wav,
                            language=language_code
                        )
                    else:
                        # Use default speaker
                        # For default speaker, we still use the regular tts method
                        wav = model.tts(
                            text=text_to_process,
                            language=language_code
                        )
                        # Save the generated audio
                        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                        sf.write(output_file, wav, 22050)  # Always use optimal sample rate
                    
                    # Add to session state for cleanup
                    st.session_state.temp_files.append(output_file)
                    
                    generation_time = time.time() - start_time
                    logger.info(f"Voice generation completed in {generation_time:.2f} seconds")
                    
                    # Verify the output file exists
                    validate_file_exists(output_file, "Generated audio file")
                    
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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"voice_clone_{timestamp}.wav"
                    
                    with open(output_file, "rb") as file:
                        st.download_button(
                            label="Download Generated Voice",
                            data=file,
                            file_name=download_filename,
                            mime="audio/wav"
                        )
                    # Close the output section with a visible footer
                    st.markdown("""
                    <div style="text-align: right; font-size: 0.8em; color: #888; margin-top: 10px;">
                      Generated voice ready for download and use
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    error_msg = f"Error generating voice: {str(e)}"
                    log_error(error_msg)
                    st.error(error_msg)

    # THIRD ROW - Tips Section
    st.markdown("<div class='row-title' style='margin-top: 60px;'>Tips for Better Results</div>", unsafe_allow_html=True)

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

    # Display app info
    st.markdown("---")
    # Footer
    st.markdown("<div style='text-align: center; color: #888; padding: 20px;'>Built with ❤️ by Principia Team</div>", unsafe_allow_html=True)
    
    st.markdown("### About")
    st.markdown("""
    This application uses YourTTS, a zero-shot multi-speaker TTS model, to clone voices from audio samples.
    
    **Features:**
    - Upload any voice recording to clone
    - Supports multiple languages
    - Advanced audio preprocessing for optimal results
    - Voice style matching to replicate speaking rate and pattern
    - GPU-accelerated inference for faster generation
    - Download generated voices for later use
    
    **Technical Details:**
    - Model: YourTTS (Coqui TTS)
    - Audio processing: High-pass filtering, silence trimming, normalization
    - Voice analysis: Speaking rate detection, pitch analysis
    - Optimal sample rate: 22050 Hz
    """)

if __name__ == "__main__":
    main() 