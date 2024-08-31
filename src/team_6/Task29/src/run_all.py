import subprocess
import os

def run_script(script_name):
    """Run a Python script using the virtual environment's Python interpreter."""
    venv_python = "venv/bin/python"  # Use a relative path to the Python interpreter in the virtual environment
    script_path = os.path.join(os.path.dirname(__file__), script_name)  # Construct the relative path to the script
    result = subprocess.run([venv_python, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def run_script_with_live_output(script_name):
    """Run a Python script and display live output."""
    venv_python = "venv/bin/python"  # Use a relative path to the Python interpreter in the virtual environment
    script_path = os.path.join(os.path.dirname(__file__), script_name)  # Construct the relative path to the script
    process = subprocess.Popen([venv_python, script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end='')  # Print each line as it comes

    process.wait()  # Wait for the process to complete

def main():
    # List of scripts to run in order
    scripts = [
        "data_cleaning.py",
        "data_processing.py",
        "model_training.py"
    ]

    # Run each script
    for script in scripts:
        print(f"Running {script}...")
        if script == "model_training.py":
            run_script_with_live_output(script)  # Show live output for model training
        else:
            run_script(script)
        print(f"Finished {script}\n")

if __name__ == "__main__":
    main()