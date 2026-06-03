#!/bin/bash

# Unix/Linux/macOS bash script to start the FastAPI backend

cd Backend

# Check if venv exists, create if it doesn't
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Start the server
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo ""
python main.py
