@echo off
REM One-click launcher for Windows: double-click this file (or run it in a terminal).
REM   1. finds Python 3.10+                     2. creates a private virtual environment in .venv\
REM   3. installs requirements.txt (first run)  4. starts the FastAPI app and opens your browser
setlocal
cd /d "%~dp0"
if "%PORT%"=="" set PORT=8000

REM --- 1. find Python -----------------------------------------------------------
set PYTHON=
where py >nul 2>nul && set PYTHON=py -3
if "%PYTHON%"=="" ( where python >nul 2>nul && set PYTHON=python )
if "%PYTHON%"=="" (
  echo ERROR: Python was not found.
  echo Install Python 3.10 or newer from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run this file again.
  pause
  exit /b 1
)
%PYTHON% -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.10 or newer is required.
  %PYTHON% --version
  pause
  exit /b 1
)

REM --- 2. virtual environment ----------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv\ ...
  %PYTHON% -m venv .venv
  if errorlevel 1 ( echo ERROR: could not create the virtual environment. & pause & exit /b 1 )
)

REM --- 3. dependencies (first run only) ------------------------------------------
if not exist ".venv\.requirements-installed" (
  echo Installing dependencies ^(first run takes a few minutes: PyTorch is a large download^) ...
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
  ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
  if errorlevel 1 ( echo ERROR: installing the dependencies failed - check your internet connection. & pause & exit /b 1 )
  copy /y requirements.txt ".venv\.requirements-installed" >nul
)

REM --- 4. start the app ----------------------------------------------------------
if not exist "models\best_model.pth" echo WARNING: models\best_model.pth is missing - run notebooks 01-07 first.
echo.
echo Starting the dashboard at http://localhost:%PORT%   ^(press Ctrl+C to stop^)
if "%NO_BROWSER%"=="" start "" /b cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:%PORT%"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
pause
