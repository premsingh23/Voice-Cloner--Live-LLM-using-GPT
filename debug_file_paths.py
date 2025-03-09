import os
import sys
import tempfile
import logging
from utils import validate_file_exists, ensure_sample_rate, convert_audio_for_model, generate_spectrogram
from utils import check_ffmpeg, cleanup_temp_files, save_audio
import numpy as np

# Configure more detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("debug_script")

def test_temp_directory():
    """Test that the temporary directory is accessible and writable"""
    logger.info("Testing temporary directory...")
    
    temp_dir = tempfile.gettempdir()
    logger.info(f"Temporary directory: {temp_dir}")
    
    if not os.path.exists(temp_dir):
        logger.error(f"Temporary directory does not exist: {temp_dir}")
        return False
    
    if not os.access(temp_dir, os.W_OK):
        logger.error(f"Temporary directory is not writable: {temp_dir}")
        return False
    
    # Try to create a file in the temp directory
    try:
        test_file = os.path.join(temp_dir, "voice_cloner_test.txt")
        with open(test_file, 'w') as f:
            f.write("Test")
        logger.info(f"Successfully created test file: {test_file}")
        
        # Clean up
        os.remove(test_file)
        logger.info(f"Successfully removed test file: {test_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to write to temporary directory: {str(e)}")
        return False

def test_audio_file(file_path):
    """Test operations on a specific audio file"""
    logger.info(f"Testing audio file: {file_path}")
    
    try:
        # Check if file exists
        validate_file_exists(file_path, "Test audio file")
        logger.info(f"File exists: {file_path}")
        
        # Test conversion
        logger.info("Testing audio conversion...")
        converted_file = convert_audio_for_model(file_path, 22050, True, True)
        logger.info(f"Conversion successful: {converted_file}")
        
        # Test spectrogram generation
        logger.info("Testing spectrogram generation...")
        spectrogram = generate_spectrogram(converted_file)
        logger.info("Spectrogram generation successful")
        
        # Test cleanup
        logger.info(f"Testing file cleanup...")
        cleanup_temp_files(converted_file)
        logger.info("Cleanup successful")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def test_audio_save():
    """Test saving audio data to a file"""
    logger.info("Testing audio save functionality...")
    
    try:
        # Create a simple sine wave
        sample_rate = 22050
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save the audio
        output_file = save_audio(audio_data, sample_rate)
        logger.info(f"Successfully saved test audio to: {output_file}")
        
        # Verify the file exists
        validate_file_exists(output_file, "Output audio file")
        logger.info(f"Verified file exists: {output_file}")
        
        # Clean up
        cleanup_temp_files(output_file)
        logger.info(f"Cleanup successful")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting file handling diagnostics...")
    
    # Test FFmpeg
    if check_ffmpeg():
        logger.info("FFmpeg is available")
    else:
        logger.warning("FFmpeg is not available - audio conversion may fail")
    
    # Test temporary directory
    if not test_temp_directory():
        logger.error("Temporary directory tests failed")
        return False
    
    # Test audio save
    if not test_audio_save():
        logger.error("Audio save tests failed")
        return False
    
    # Ask user for a test file
    print("\nWould you like to test with a specific audio file? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        print("Enter the full path to an audio file:")
        file_path = input().strip()
        
        if os.path.exists(file_path):
            test_audio_file(file_path)
        else:
            logger.error(f"File not found: {file_path}")
    
    logger.info("Diagnostics completed")
    return True

if __name__ == "__main__":
    main() 