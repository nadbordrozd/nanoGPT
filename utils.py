from pathlib import Path
import logging


BASE_PATH = Path(__file__).resolve().parent
DATA_DIR = BASE_PATH / 'data'
CONFIG_DIR = BASE_PATH / 'config'
MODELS_DIR = BASE_PATH / 'models'

def get_validation_text(dataset):
    with open(DATA_DIR / dataset / 'val.txt', 'r') as f:
        return f.read()


logger = logging.getLogger('nanoGPT')
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
