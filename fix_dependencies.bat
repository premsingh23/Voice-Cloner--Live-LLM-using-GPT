@echo off
echo =======================================
echo Voice Cloner - Dependency Fix Script
echo =======================================
echo.

echo This script will fix the dependencies to ensure proper functionality.
echo It will fix the librosa display issue and improve voice cloning quality.
echo.

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /B 1
)
echo [OK] Virtual environment activated.
echo.

:: Install/update the required packages
echo Installing dependencies...
echo.

echo 1. Checking current package versions...
pip freeze
echo.

echo 2. Uninstalling conflicting packages...
pip uninstall -y librosa matplotlib
echo.

echo 3. Installing compatible matplotlib version...
pip install matplotlib==3.5.3
echo.

echo 4. Installing compatible librosa version...
pip install librosa==0.9.2
echo.

echo 5. Installing TTS dependencies for better voice cloning...
pip install numba>=0.51.0 resampy>=0.2.2 SoundFile>=0.10.3
echo.

echo 6. Reinstalling PyTorch (if needed)...
pip install torch==1.13.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
echo.

echo 7. Checking if librosa.display is available...
python -c "import librosa.display; print('Success: librosa.display is available')" || echo "Error: librosa.display is still not available"
echo.

echo 8. Testing matplotlib compatibility...
python -c "import matplotlib.pyplot as plt; import numpy as np; plt.figure(); plt.imshow(np.random.rand(10,10)); plt.close(); print('Success: matplotlib is working correctly')" || echo "Error: matplotlib test failed"
echo.

echo 9. Installing additional libraries for better voice quality...
pip install praat-parselmouth==0.4.3
echo.

echo All dependencies have been updated.
echo You should now run the application again using run_app_only.bat
echo.

:: Deactivate virtual environment
deactivate
pause 