import subprocess

import yaml

# Load configuration from the specified file
config_path = 'configs/main_config.yml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

    # Extract specific configuration values
    BUILD_DATASET = config['BUILD_DATASET']
    BUILD_MODEL = config['BUILD_MODEL']
    USE_SELECTION_IF_AVAILABLE = config['USE_SELECTION_IF_AVAILABLE']

# Print confirmation
print(f"Configuration loaded from {config_path}")


def main():
    # Path to the virtual environment's Python interpreter
    venv_python = r"Scripts\python.exe"


    # Execute the dataset creation script only when BUIL_DATASET is true
    if BUILD_DATASET:
        subprocess.run([venv_python, "./src/1_create_dataset.py"], check=True)
    else:
        print("Dataset creation skipped.")
    
    # Execute the first script using the venv Python
    if BUILD_MODEL:
        subprocess.run([venv_python, "./src/2_modeling.py"], check=True)
    else:
        print("Model building skipped.")
    
    # Execute the second script using the venv Python
    subprocess.run([venv_python, "./src/3_backtesting.py"], check=True)


if __name__ == "__main__":
    main()
