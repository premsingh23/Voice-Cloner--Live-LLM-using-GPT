# Voice Cloner

A voice cloning application built with YourTTS and Streamlit. This application allows you to clone any voice using just a short audio sample.

## Features

- Upload a sample voice recording
- Input text to be spoken in the cloned voice
- Generate speech that sounds like the uploaded voice
- Display spectrograms of both the input and output audio
- Support for multiple languages
- GPU acceleration with CUDA

## Technical Requirements

- Windows 11
- Python 3.9 (exactly 3.9, other versions may cause compatibility issues)
- NVIDIA GPU with CUDA support (recommended)
- FFmpeg (required for audio conversion)

## Installation

### Option 1: Using the provided batch script

1. Make sure you have Python 3.9 installed and added to your PATH
2. Make sure FFmpeg is installed and added to your PATH
3. Double-click `run_app.bat` to automatically set up the environment and launch the application

### Option 2: Manual installation

1. Make sure you have Python 3.9 installed and added to your PATH
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Install PyTorch with CUDA support:
   ```
   pip install torch==1.13.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```
6. Install TTS from GitHub:
   ```
   pip install git+https://github.com/coqui-ai/TTS.git@v0.12.0
   ```
7. Launch the application:
   ```
   streamlit run app.py
   ```

## CUDA Setup

For optimal performance, this application requires CUDA Toolkit 11.6, which is compatible with PyTorch 1.13.1:

1. Download CUDA Toolkit 11.6 from the [NVIDIA archives](https://developer.nvidia.com/cuda-11-6-0-download-archive)
2. Follow the installation instructions
3. Verify CUDA installation by running `check_cuda.py`:
   ```
   python check_cuda.py
   ```

## Usage

1. Launch the application using the batch script or by running `streamlit run app.py`
2. Upload a voice sample (WAV, MP3, or OGG format)
3. Enter the text you want to convert to speech
4. Click "Generate Voice" to create the cloned voice
5. Listen to the result and view the spectrograms

## Troubleshooting

- **Audio conversion errors**: Make sure FFmpeg is installed and added to your PATH
- **CUDA errors**: Verify that you have CUDA Toolkit 11.6 installed and an NVIDIA GPU
- **Package conflicts**: Use the exact package versions specified in `requirements.txt`
- **TTS model errors**: Check that you're using the correct version of TTS (v0.12.0)

## File Structure

- `app.py`: Main Streamlit application
- `utils.py`: Helper functions for audio processing
- `check_cuda.py`: Script to verify CUDA availability
- `run_app.bat`: Batch script to set up the environment and run the app
- `requirements.txt`: List of required Python packages with versions
 
