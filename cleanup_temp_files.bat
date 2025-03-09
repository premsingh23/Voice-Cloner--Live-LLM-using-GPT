@echo off
echo =======================================
echo Voice Cloner - Temporary File Cleanup
echo =======================================
echo.

echo This script will clean up temporary files created by the Voice Cloner application.
echo These files may be left behind if the application doesn't exit cleanly.
echo.

set TEMP_DIR=%TEMP%
echo Temporary directory: %TEMP_DIR%
echo.

echo Searching for Voice Cloner temporary files...
echo.

set FILE_COUNT=0
set FILE_SIZE=0

for /f "tokens=*" %%G in ('dir /b /s "%TEMP_DIR%\voice_cloner_*" 2^>nul') do (
    echo Found: %%G
    set /a FILE_COUNT+=1
    for %%A in ("%%G") do set /a FILE_SIZE+=%%~zA
)

echo.
if %FILE_COUNT% EQU 0 (
    echo No Voice Cloner temporary files found.
) else (
    echo Found %FILE_COUNT% temporary files.
    set /a FILE_SIZE_KB=%FILE_SIZE% / 1024
    echo Total size: approximately %FILE_SIZE_KB% KB
    echo.
    
    set /p CONFIRM=Do you want to delete these files? (Y/N): 
    if /i "%CONFIRM%"=="Y" (
        echo Deleting files...
        for /f "tokens=*" %%G in ('dir /b /s "%TEMP_DIR%\voice_cloner_*" 2^>nul') do (
            echo Deleting: %%G
            del "%%G" 2>nul
        )
        echo Cleanup complete.
    ) else (
        echo Cleanup cancelled.
    )
)

echo.
echo Also checking for upload_ temporary files...
echo.

set UPLOAD_COUNT=0
set UPLOAD_SIZE=0

for /f "tokens=*" %%G in ('dir /b /s "%TEMP_DIR%\upload_*" 2^>nul') do (
    echo Found: %%G
    set /a UPLOAD_COUNT+=1
    for %%A in ("%%G") do set /a UPLOAD_SIZE+=%%~zA
)

echo.
if %UPLOAD_COUNT% EQU 0 (
    echo No upload temporary files found.
) else (
    echo Found %UPLOAD_COUNT% upload temporary files.
    set /a UPLOAD_SIZE_KB=%UPLOAD_SIZE% / 1024
    echo Total size: approximately %UPLOAD_SIZE_KB% KB
    echo.
    
    set /p CONFIRM=Do you want to delete these files? (Y/N): 
    if /i "%CONFIRM%"=="Y" (
        echo Deleting files...
        for /f "tokens=*" %%G in ('dir /b /s "%TEMP_DIR%\upload_*" 2^>nul') do (
            echo Deleting: %%G
            del "%%G" 2>nul
        )
        echo Cleanup complete.
    ) else (
        echo Cleanup cancelled.
    )
)

echo.
echo Done.
pause 