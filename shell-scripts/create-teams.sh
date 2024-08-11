#!/bin/bash

# Specify the number of folders to create
number_of_folders=11

# Loop to create each folder and file
for (( i=1; i<=number_of_folders; i++ ))
do
    folder_name="src/team_$i"
    test_folder_name="tests/team_$i"

    # Check if the team folder exists in src
    if [ ! -d "$folder_name" ]; then
        # Create a new directory
        mkdir "$folder_name"    
        # Create an empty __init__.py file in the newly created directory
        touch "src/team_$i/__init__.py"
        echo "Created directory: $folder_name"
    else
        echo "Directory $folder_name already exists, skipping creation."
    fi

    # Check if the main folder exists
    if [ ! -d "$test_folder_name" ]; then
        # Create a new directory
        mkdir "$test_folder_name"    
        # Create an empty __init__.py file in the newly created directory
        touch "tests/team_$i/__init__.py"
        echo "Created directory: $test_folder_name"
    else
        echo "Directory $test_folder_name already exists, skipping creation."
    fi
done

echo "Team Folders and __init__.py files created successfully."
