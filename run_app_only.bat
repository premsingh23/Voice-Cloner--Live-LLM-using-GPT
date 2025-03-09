@echo off
echo =======================================
echo Voice Cloner - Quick Launch
echo =======================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

if not exist "app.py" (
    echo [ERROR] app.py not found in the current directory. 
    echo Current directory: %CD%
    goto END
)

echo Launching Streamlit application...
echo If the application doesn't start in your browser, go to:
echo    http://localhost:8501
echo.
echo (Press Ctrl+C to stop the application when done)
echo.

streamlit run app.py

:END
deactivate
echo.
pause