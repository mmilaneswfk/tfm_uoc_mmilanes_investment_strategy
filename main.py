import subprocess

def main():
    # Path to the virtual environment's Python interpreter
    venv_python = r"Scripts\python.exe"

    # Execute the first script using the venv Python
    subprocess.run([venv_python, "./src/classification_modeling_single.py"], check=True)
    
    # Execute the second script using the venv Python
    subprocess.run([venv_python, "./src/classification_backtesting_single.py"], check=True)

if __name__ == "__main__":
    main()
