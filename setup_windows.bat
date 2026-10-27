@echo off
rem MotionPlay one-time setup for Windows. Double-click it, or run it from any
rem terminal. Batch files are not blocked by PowerShell's script policy.
rem   setup_windows.bat          -> DirectML (any DirectX 12 GPU, easiest)
rem   setup_windows.bat nvidia   -> CUDA/TensorRT build for NVIDIA cards
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer.
  goto :fail
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo MotionPlay needs Python 3.10 or newer.
  goto :fail
)

set "RUNTIME=directml"
if /i "%~1"=="nvidia" set "RUNTIME=gpu"

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating a private Python environment in .venv ...
  python -m venv .venv
  if errorlevel 1 goto :fail
)

echo [2/4] Installing MotionPlay with the %RUNTIME% runtime ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>nul
".venv\Scripts\python.exe" -m pip install -e ".[%RUNTIME%,mediapipe]"
if errorlevel 1 goto :fail

echo [3/4] Downloading models ...
".venv\Scripts\python.exe" -m motionplay models download

echo [4/4] Measuring this PC to pick a quality level ...
".venv\Scripts\python.exe" -m motionplay bench

echo.
echo Setup finished. Double-click start_windows.bat to play.
pause
exit /b 0

:fail
echo.
echo Setup stopped - read the message above.
pause
exit /b 1
