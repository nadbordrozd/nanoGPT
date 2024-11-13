from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
DATA_DIR = BASE_PATH / 'data'
CONFIG_DIR = BASE_PATH / 'config'
MODELS_DIR = BASE_PATH / 'models'

def get_validation_text(dataset):
    with open(DATA_DIR / dataset / 'val.txt', 'r') as f:
        return f.read()
