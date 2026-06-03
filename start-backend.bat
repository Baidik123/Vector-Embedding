@echo off
REM Windows batch script to start the FastAPI backend

cd Backend

REM Check if venv exists, create if it doesn't
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

REM Start the server
echo.
echo Starting FastAPI server on http://localhost:8000
echo.
python main.py
