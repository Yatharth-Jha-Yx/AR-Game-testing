@echo off
rem Start the MotionPlay engine and open the launcher in your browser.
rem Extra options are passed through, e.g.:
rem   start_windows.bat --source synthetic:demo     (no camera needed)
rem   start_windows.bat --source camera:1           (second webcam)
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
echo Starting MotionPlay. Keep this window open while you play; close it to stop.
".venv\Scripts\python.exe" -m motionplay run --open %*
pause
