#!/bin/bash

# Function to determine the correct Python command
find_python() {
    # First, check if 'python' is available and linked to Python 3
    if command -v python > /dev/null 2>&1; then
        PYTHON_VERSION=$(python --version 2>&1 | grep -o 'Python [0-9]' | grep -o '[0-9]')
        if [ "$PYTHON_VERSION" -ge 3 ]; then
            echo "Using python..."
            PYTHON_CMD=python
        else
            echo "Python 3 is not installed on 'python', checking 'python3'..."
            check_python3
        fi
    else
        echo "'python' command is not available, checking 'python3'..."
        check_python3
    fi
}

# Helper function to check for python3 if python is not suitable
check_python3() {
    if command -v python3 > /dev/null 2>&1; then
        echo "Using python3..."
        PYTHON_CMD=python3
    else
        echo "Neither 'python' nor 'python3' point to Python 3. Exiting."
        exit 1
    fi
}

# Detect the correct Python command
find_python

# Create a new Python virtual environment named 'venv'
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv
echo "Virtual environment created."

# Function to activate the virtual environment
activate_env() {
    # Determine the platform-specific activation script
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        VENV_ACTIVATE="venv/bin/activate"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        VENV_ACTIVATE="venv/Scripts/activate"
    else
        echo "Unsupported OS or shell. Exiting."
        exit 1
    fi
    
    # Attempt to source the activation script
    echo "Activating virtual environment..."
    if source $VENV_ACTIVATE; then
        echo "Virtual environment activated."
    else
        echo "Failed to activate virtual environment. Exiting."
        exit 1
    fi
}

# Activate the virtual environment
activate_env

# Check and install requirements if requirements.txt exists
if [ -f requirements.txt ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    echo "Requirements installed."
else
    echo "requirements.txt not found."
fi

echo "Setup completed."
