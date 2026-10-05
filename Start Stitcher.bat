@echo off
rem Double-click this file to open the Whisky Webs Stitcher.
cd /d "%~dp0"

set "PY="
python --version >nul 2>nul && set "PY=python" && set "PYW=pythonw"
if not defined PY py -3 --version >nul 2>nul && set "PY=py -3" && set "PYW=pyw -3"
if not defined PY (
    echo Python is not installed on this computer.
    echo Install it from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^),
    echo then double-click this file again.
    pause
    exit /b 1
)

%PY% -c "import numpy, PIL, tifffile, imagecodecs, customtkinter, cv2, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo First run: installing the parts the stitcher needs. This only happens once...
    %PY% -m pip install --user -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installation failed - please send a screenshot of this window to whoever set this up.
        pause
        exit /b 1
    )
)

start "" %PYW% stitch_gui.py
