#!/bin/bash

# Check if an integer parameter was provided
if [[ $# -eq 0 ]] || ! [[ $1 =~ ^[0-9]+$ ]]; then
    echo "Please provide an integer parameter."
    exit 1
fi

# Directory containing the source code, including the team subdirectory with integer suffix
SRC_DIR="src/team_$1"

# Check if the source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "The directory $SRC_DIR does not exist."
    exit 1
fi

# Apply isort to sort imports
echo "Running isort on $SRC_DIR..."
isort $SRC_DIR

# Apply black to format the code
echo "Running black on $SRC_DIR..."
black $SRC_DIR

echo "Formatting complete."
