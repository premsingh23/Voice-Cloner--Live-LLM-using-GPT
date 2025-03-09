@echo off
SETLOCAL EnableDelayedExpansion

:: Set window title
TITLE Voice Cloner - Direct Launcher

echo =======================================
echo Voice Cloner - Direct Launch Script
echo =======================================
echo.

echo This script will directly use the Python launcher or allow you to specify the Python path.
echo.

:SELECT_PYTHON
echo Select an option to launch the app:
echo 1. Use Python Launcher (py -3.9) - recommended if you have Python launcher installed
echo 2. Use python command (if Python 3.9 is your default Python version)
echo 3. Use python3 command (if you have multiple Python versions)
echo 4. Specify full path to Python 3.9 executable
echo.

set /p CHOICE="Enter your choice (1-4): "

if "%CHOICE%"=="1" (
    set PYTHON_CMD=py -3.9
    goto CHECK_VERSION
) else if "%CHOICE%"=="2" (
    set PYTHON_CMD=python
    goto CHECK_VERSION
) else if "%CHOICE%"=="3" (
    set PYTHON_CMD=python3
    goto CHECK_VERSION
) else if "%CHOICE%"=="4" (
    echo Please enter the full path to your Python 3.9 executable
    echo Example: C:\Users\username\AppData\Local\Programs\Python\Python39\python.exe
    echo.
    set /p PYTHON_PATH="Python path: "
    set PYTHON_CMD="%PYTHON_PATH%"
    goto CHECK_VERSION
) else (
    echo Invalid choice. Please select 1-4.
    goto SELECT_PYTHON
)

:CHECK_VERSION
echo.
echo Checking Python version...
%PYTHON_CMD% --version
echo.
echo If the version above is not Python 3.9.x, press Ctrl+C to abort and try again.
echo Otherwise, press any key to continue.
pause > nul

:: Check for virtual environment
echo.
echo Checking for virtual environment...
if not exist "venv\" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /B 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment exists.
)
echo.

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /B 1
)
echo [OK] Virtual environment activated.
echo.

:: Install dependencies if needed
echo Checking for required packages...
pip show streamlit >NUL 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo Required packages are not installed. Installing dependencies...
    echo This may take a while...
    
    echo Installing base dependencies...
    pip install numpy==1.22.4 pandas==1.5.3 matplotlib==3.7.2 scipy==1.10.1 
    pip install torch==1.13.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install protobuf==3.20.3 streamlit==1.24.0
    pip install librosa>=0.9.2 pydub>=0.25.1 soundfile>=0.12.1
    
    echo Installing TTS from GitHub...
    pip install git+https://github.com/coqui-ai/TTS.git@v0.12.0
    
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to install required packages.
        deactivate
        pause
        exit /B 1
    )
    echo [OK] All packages installed successfully.
) else (
    echo [OK] Required packages are already installed.
)
echo.

:: Check for FFmpeg
echo Checking for FFmpeg...
where ffmpeg >NUL 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] FFmpeg is not found in the PATH.
    echo Audio conversions may not work properly without FFmpeg.
    echo.
    echo Would you like to continue anyway?
    choice /C YN /M "Continue without FFmpeg? [Y/N]"
    if !ERRORLEVEL! EQU 2 (
        echo Aborting setup.
        deactivate
        pause
        exit /B 1
    )
) else (
    echo [OK] FFmpeg is available.
)
echo.

:: Check if app.py exists
if not exist "app.py" (
    echo [ERROR] app.py not found in the current directory.
    echo Make sure you're running this script from the root directory of the Voice Cloner project.
    echo Current directory: %CD%
    echo.
    echo Available files in current directory:
    dir /b
    echo.
    deactivate
    pause
    exit /B 1
)

:: Run the application
echo =======================================
echo All checks passed! Starting Voice Cloner...
echo =======================================
echo.

echo Launching Streamlit application...
echo If the application doesn't open in your browser automatically, go to:
echo    http://localhost:8501
echo.

:: Try running Streamlit with explicit output
streamlit run app.py
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Failed to run Streamlit application.
    echo.
    echo Trying alternative method...
    echo.
    python -m streamlit run app.py
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to launch application.
        echo.
        echo Debugging information:
        echo - Current directory: %CD%
        echo - Streamlit version:
        pip show streamlit
        echo.
        echo - Python version:
        python --version
        echo.
        echo You can try running the app manually by:
        echo 1. Making sure the virtual environment is activated (venv\Scripts\activate)
        echo 2. Running: streamlit run app.py
        echo.
    )
)

:: Deactivate virtual environment when done
deactivate
echo.
echo Application has been closed. Thank you for using Voice Cloner.
echo.
pause 