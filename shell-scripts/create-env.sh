#!/bin/bash

# Function to determine the correct Python command
find_python() {
    if command -v python > /dev/null 2>&1; then
        PYTHON_CMD=python
    elif command -v python3 > /dev/null 2>&1; then
        PYTHON_CMD=python3
    else
        echo "Python is not installed or not found in PATH. Exiting."
        exit 1
    fi
}

# Detect the correct Python command
find_python

# Create a new Python virtual environment named 'venv'
$PYTHON_CMD -m venv venv

# Function to activate the virtual environment
activate_env() {
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        source venv/bin/activate
        # Check if we're now using the Python from the virtual environment (Unix-like systems)
        if [[ "$(pwd)/venv/bin/python" != "$(which python)" ]]; then
            echo "Failed to activate virtual environment properly on Unix. Exiting."
            exit 1
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
        # Check if we're now using the Python from the virtual environment (Windows)
        local win_python_path="$(echo $VIRTUAL_ENV | sed -e 's/\\/\//g')/Scripts/python"
        local where_python
        where_python=$(where python | grep -i "$(echo $PWD | sed -e 's/\\/\//g')/venv/Scripts")
        if [[ "$where_python" != "$win_python_path" ]]; then
            echo "Failed to activate virtual environment properly on Windows. Exiting."
            exit 1
        fi
    else
        echo "Unsupported OS or shell. Exiting."
        exit 1
    fi
}

# Activate the virtual environment
activate_env

# Install requirements if requirements.txt exists
if [ -f requirements.txt ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi

echo "Setup completed."
