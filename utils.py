import os
import tempfile
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
import torch
import io
import base64
from scipy import signal
import time
import random
import string
import shutil
import logging

# Import librosa.display explicitly
try:
    import librosa.display
except ImportError:
    logging.warning("librosa.display could not be imported. Spectrograms will not be available.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_cloner")

def get_unique_filename(suffix='.wav'):
    """
    Generate a unique filename with a random string to avoid conflicts.
    
    Args:
        suffix (str): File extension
        
    Returns:
        str: Unique filename
    """
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    timestamp = int(time.time())
    return f"voice_cloner_{timestamp}_{random_str}{suffix}"

def retry_on_error(func, max_retries=3, retry_delay=1.0):
    """
    Retry a function if it fails with a file access error.
    
    Args:
        func: Function to retry
        max_retries (int): Maximum number of retry attempts
        retry_delay (float): Delay between retries in seconds
        
    Returns:
        The result of the function call
    """
    for attempt in range(max_retries):
        try:
            return func()
        except (PermissionError, OSError) as e:
            # Don't retry if file doesn't exist - that won't help
            if isinstance(e, FileNotFoundError) or "No such file" in str(e) or "cannot find the file" in str(e):
                logger.error(f"File not found error: {str(e)}. Cannot retry as file doesn't exist.")
                raise
            
            if attempt == max_retries - 1:
                logger.error(f"Max retries ({max_retries}) reached for operation. Giving up.")
                raise
                
            logger.info(f"File access error: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay + random.random())  # Add randomness to prevent collisions

def validate_file_exists(file_path, description="File"):
    """
    Validate that a file exists and is accessible.
    
    Args:
        file_path (str): Path to the file
        description (str): Description of the file for error messages
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not file_path:
        raise FileNotFoundError(f"{description} path is empty or None")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found at: {file_path}")
        
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{description} exists but is not a file: {file_path}")
    
    return True

def check_ffmpeg():
    """
    Check if FFmpeg is available in PATH or at a known Windows location.
    
    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    # First check PATH
    ffmpeg_path = shutil.which("ffmpeg")
    
    # If not found in PATH, check specific Windows location
    if not ffmpeg_path and os.path.exists("C:\\FFmpeg\\bin\\ffmpeg.exe"):
        ffmpeg_path = "C:\\FFmpeg\\bin\\ffmpeg.exe"
        # Set environment variable for this session
        os.environ["PATH"] += os.pathsep + "C:\\FFmpeg\\bin"
        return True
        
    return ffmpeg_path is not None

def check_cuda_availability():
    """
    Check if CUDA is available and return device information.
    
    Returns:
        tuple: (device, device_info_dict)
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        device_info = {
            "device": device,
            "name": device_name,
            "count": device_count,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        }
    else:
        device = "cpu"
        device_info = {
            "device": device
        }
    
    return device, device_info

def ensure_sample_rate(audio_file, sample_rate=22050, trim_silence=False, normalize=False):
    """
    Ensure the audio file has the correct sample rate and apply optional preprocessing.
    
    Args:
        audio_file (str): Path to the audio file
        sample_rate (int): Target sample rate
        trim_silence (bool): Whether to trim silence from the audio
        normalize (bool): Whether to normalize the audio
        
    Returns:
        str: Path to the audio file with the correct sample rate and preprocessing
    """
    # Validate input file exists
    try:
        validate_file_exists(audio_file, "Input audio file")
    except FileNotFoundError as e:
        logger.error(f"Input validation error: {str(e)}")
        raise
    
    # Create a temporary file with a unique name to store the resampled audio
    temp_dir = tempfile.gettempdir()
    unique_filename = get_unique_filename(suffix='.wav')
    temp_path = os.path.join(temp_dir, unique_filename)
    
    logger.info(f"Processing audio file: {audio_file}")
    logger.info(f"Output will be saved to: {temp_path}")
    
    try:
        # If trim_silence or normalize are requested, use librosa for advanced processing
        if trim_silence or normalize:
            def process_with_librosa():
                # Load audio
                logger.info(f"Loading audio with librosa: {audio_file}")
                y, sr = librosa.load(audio_file, sr=None)
                
                # Trim silence if requested
                if trim_silence:
                    logger.info("Trimming silence")
                    y, _ = librosa.effects.trim(y, top_db=20)
                
                # Resample if necessary
                if sr != sample_rate:
                    logger.info(f"Resampling from {sr} to {sample_rate}")
                    y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                
                # Normalize if requested
                if normalize:
                    logger.info("Normalizing audio")
                    y = librosa.util.normalize(y)
                
                # Save processed audio
                logger.info(f"Saving processed audio to: {temp_path}")
                sf.write(temp_path, y, sample_rate)
                return temp_path
            
            return retry_on_error(process_with_librosa)
        else:
            # Use pydub for basic conversion (faster)
            def load_and_export():
                logger.info(f"Loading audio with pydub: {audio_file}")
                audio = AudioSegment.from_file(audio_file)
                logger.info(f"Setting frame rate to: {sample_rate}")
                audio = audio.set_frame_rate(sample_rate)
                logger.info(f"Exporting to: {temp_path}")
                audio.export(temp_path, format="wav")
                
                # Verify the file was created and is accessible
                validate_file_exists(temp_path, "Processed audio file")
                return temp_path
            
            return retry_on_error(load_and_export)
    except Exception as e:
        # Clean up the temp file if it exists
        if os.path.exists(temp_path):
            try:
                logger.info(f"Cleaning up temp file after error: {temp_path}")
                os.unlink(temp_path)
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                pass
        logger.error(f"Error processing audio: {str(e)}")
        raise Exception(f"Error processing audio: {str(e)}")

def convert_audio_for_model(audio_file, model_sample_rate=22050, trim_silence=True, normalize=True):
    """
    Convert audio file to the correct format for the TTS model with preprocessing.
    
    Args:
        audio_file (str): Path to the audio file
        model_sample_rate (int): The sample rate required by the model
        trim_silence (bool): Whether to trim silence from the audio
        normalize (bool): Whether to normalize the audio
        
    Returns:
        str: Path to the audio file in the correct format
    """
    # Apply preprocessing and ensure correct sample rate
    return ensure_sample_rate(
        audio_file, 
        sample_rate=model_sample_rate,
        trim_silence=trim_silence,
        normalize=normalize
    )

def generate_spectrogram(audio_file, title="Spectrogram"):
    """
    Generate and return a spectrogram of the audio file.
    
    Args:
        audio_file (str): Path to the audio file
        title (str): Title for the spectrogram
        
    Returns:
        str: Base64 encoded image of the spectrogram
    """
    try:
        # Check if librosa.display is available
        if not hasattr(librosa, 'display'):
            raise ImportError("librosa.display module is not available. Please install with 'pip install librosa>=0.9.2'")
        
        # Validate input file exists
        validate_file_exists(audio_file, "Audio file for spectrogram")
        
        # Load the audio file
        def load_audio():
            logger.info(f"Loading audio for spectrogram: {audio_file}")
            y, sr = librosa.load(audio_file, sr=None)
            return y, sr
            
        y, sr = retry_on_error(load_audio)
        
        # Create figure and axes
        plt.figure(figsize=(10, 4))
        
        # Generate a spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        
        # Add color bar and title
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        # Save the image to a memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encode the image to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close the plot
        plt.close()
        
        return img_str
    except ImportError as e:
        logger.error(f"Import error for spectrograms: {str(e)}")
        raise Exception(f"Error generating spectrogram: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating spectrogram: {str(e)}")
        raise Exception(f"Error generating spectrogram: {str(e)}")

def get_duration(audio_file):
    """
    Get the duration of an audio file.
    
    Args:
        audio_file (str): Path to the audio file
        
    Returns:
        float: Duration of the audio file in seconds
    """
    try:
        # Validate input file exists
        validate_file_exists(audio_file, "Audio file for duration")
        
        def load_audio():
            y, sr = librosa.load(audio_file, sr=None)
            return y, sr
            
        y, sr = retry_on_error(load_audio)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        logger.error(f"Error getting audio duration: {str(e)}")
        raise Exception(f"Error getting audio duration: {str(e)}")

def save_audio(waveform, sample_rate, output_file=None, normalize=True):
    """
    Save audio waveform to a file.
    
    Args:
        waveform (torch.Tensor, numpy.ndarray, or list): Audio waveform
        sample_rate (int): Sample rate of the audio
        output_file (str, optional): Path to save the audio file. If None, creates a temporary file.
        normalize (bool): Whether to normalize the audio before saving
        
    Returns:
        str: Path to the saved audio file
    """
    try:
        # Create a unique output file if none provided
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            unique_filename = get_unique_filename(suffix='.wav')
            output_file = os.path.join(temp_dir, unique_filename)
            
        logger.info(f"Saving audio to: {output_file}")
        
        # Convert to numpy array if it's a list
        if isinstance(waveform, list):
            logger.info("Converting list to numpy array")
            waveform = np.array(waveform)
        
        # Convert tensor to numpy if needed
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # Ensure the waveform is 1D
        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()
        
        # Normalize if requested
        if normalize:
            logger.info("Normalizing output audio")
            waveform = librosa.util.normalize(waveform)
        elif waveform.max() > 1.0 or waveform.min() < -1.0:
            # Still ensure the waveform is in the correct range if not normalizing
            waveform = waveform / max(abs(waveform.max()), abs(waveform.min()))
        
        # Define the save function for retry
        def write_audio():
            logger.info(f"Writing audio data to: {output_file}")
            sf.write(output_file, waveform, sample_rate)
            
            # Verify the file was created and is accessible
            validate_file_exists(output_file, "Output audio file")
            return output_file
            
        # Save the audio with retry
        return retry_on_error(write_audio)
    except Exception as e:
        logger.error(f"Error saving audio: {str(e)}")
        raise Exception(f"Error saving audio: {str(e)}")

def cleanup_temp_files(file_path):
    """
    Safely delete a temporary file if it exists.
    
    Args:
        file_path (str): Path to the file to delete
    """
    if not file_path:
        logger.warning("Attempted to clean up None or empty file path")
        return
        
    if os.path.exists(file_path):
        try:
            logger.info(f"Cleaning up file: {file_path}")
            os.unlink(file_path)
            logger.info(f"Successfully deleted: {file_path}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not delete file {file_path}: {str(e)}")
            # If we can't delete now, try to delete on program exit
            try:
                temp_dir = tempfile.gettempdir()
                # Only mark for deletion if it's in the temp directory
                if os.path.commonpath([file_path, temp_dir]) == temp_dir:
                    tempfile._get_default_tempdir()
            except Exception as ex:
                logger.warning(f"Error marking file for delayed deletion: {str(ex)}")
    else:
        logger.warning(f"File not found during cleanup: {file_path}") 