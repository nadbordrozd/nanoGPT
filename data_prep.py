import os
import pickle
import numpy as np
import tiktoken
import random
import argparse
from pathlib import Path

from torch.distributed import is_initialized

from .utils import DATA_DIR

class CharEncoder(object):
    def __init__(self, chars):
        chars = sorted(list(set(chars)))
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s, allowed_special=None):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


def encoder_from_dict(d):
    if d['encoder'] == 'char':
        return CharEncoder(d['chars'])
    else:
        return tiktoken.get_encoding('gpt2')

def encoder_to_dict(enc):
    if isinstance(enc, CharEncoder):
        return {'encoder': 'char', 'vocab_size': enc.vocab_size, 'chars': enc.chars}
    elif isinstance(enc, tiktoken.core.Encoding):
        return {'encoder': enc.name, 'vocab_size': enc.n_vocab}
    else:
        raise ValueError(f'Enc has to be either a CharEncoder or an instance of tiktoken.core.Encoding but {enc} found')


def load_encoder(dataset):
    meta_path = DATA_DIR / dataset / 'meta.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return encoder_from_dict(meta)

def prepare_char_level_data(output_dir, data, test_fraction=0.1):
    """
    Prepares data for character-level language modeling by encoding characters to integers
    and saving train, validation splits along with raw text and encoder/decoder mappings.
    
    Parameters:
    - output_dir (str): Path to the directory where output files will be saved.
    - data (str): Input text data as a string (lines separated by '\n').
    - test_fraction (float): Fraction of data to use for validation (default 0.1).
    
    Returns:
    - None: Files are saved to the specified output directory.
    """
    if not isinstance(data, str):
        raise ValueError("Input data must be a string.")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be a float between 0 and 1.")

    # Ensure output directory exists
    dataset_dir = DATA_DIR / output_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Split data into lines
    lines = data.splitlines()
    n_lines = len(lines)
    split_idx = int(n_lines * (1 - test_fraction))
    
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # Join lines back into raw text
    train_data = "\n".join(train_lines)
    val_data = "\n".join(val_lines)

    # Create character vocabulary and mappings
    chars = sorted(list(set(data)))
    enc = CharEncoder(chars)

    # Encode data to integer IDs
    train_ids = np.array(enc.encode(train_data), dtype=np.uint16)
    val_ids = np.array(enc.encode(val_data), dtype=np.uint16)

    # Save raw text
    train_raw_path = dataset_dir / 'train.txt'
    val_raw_path = dataset_dir / 'val.txt'
    with open(train_raw_path, 'w', encoding='utf-8') as f:
        f.write(train_data)
    with open(val_raw_path, 'w', encoding='utf-8') as f:
        f.write(val_data)

    # Save encoded data to binary files
    train_file_path = dataset_dir / 'train.bin'
    val_file_path = dataset_dir / 'val.bin'
    train_ids.tofile(train_file_path)
    val_ids.tofile(val_file_path)
    
    # Save metadata
    meta = encoder_to_dict(enc)
    meta_file_path = dataset_dir / 'meta.pkl'
    with open(meta_file_path, 'wb') as f:
        pickle.dump(meta, f)

    # Provide user feedback
    print(f"Data preparation complete.")
    print(f"Output directory: {dataset_dir.resolve()}")
    print(f"Vocabulary size: {enc.vocab_size}")
    print(f"Training lines: {len(train_lines):,}")
    print(f"Validation lines: {len(val_lines):,}")
    print(f"""Files saved: 
    - {train_raw_path}
    - {val_raw_path}
    - {train_file_path}
    - {val_file_path}
    - {meta_file_path}""")


def prepare_bpe_data(output_dir, input_data, test_fraction=0.1):
    """
    Processes text data given as a string, encodes it using BPE, splits it into training and validation sets,
    and saves the result in binary files.

    Parameters:
        output_dir (str): Directory where the data files will be saved.
        input_data (str): The full input text data as a string.
        test_fraction (float): Fraction of the data to be used as validation set.
    """
    # Ensure output directory exists
    dataset_dir = DATA_DIR / output_dir
    os.makedirs(dataset_dir, exist_ok=True)

    # Split data into training and validation sets
    n = len(input_data)
    train_data = input_data[:int(n * (1 - test_fraction))]
    val_data = input_data[int(n * (1 - test_fraction)):]

    # Encode with tiktoken GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Convert to numpy arrays and save as binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(dataset_dir / 'train.bin')
    val_ids.tofile(dataset_dir / 'val.bin')

    meta = encoder_to_dict(enc)
    meta_file_path = dataset_dir / 'meta.pkl'
    with open(meta_file_path, 'wb') as f:
        pickle.dump(meta, f)


def generate_equations(n, num_equations):
    """
    Generates a string of unique addition equations, each on a new line.

    Parameters:
        n (int): Number of digits for each operand.
        num_equations (int): Number of equations to generate.

    Returns:
        str: A string containing the generated equations, each on a separate line.
    """
    equations = set()  # Use a set to track unique equations
    lower_limit = 10**(n - 1)
    upper_limit = 10**n - 1

    while len(equations) < num_equations:
        a = random.randint(lower_limit, upper_limit)
        b = random.randint(lower_limit, upper_limit)
        # Ensure the pair is stored uniquely (order doesn't matter)
        if (a, b) not in equations and (b, a) not in equations:
            equations.add((a, b))  # Add the pair to the set

    # Format the equations as strings
    equation_strings = [f"{a} + {b} = {a + b}" for a, b in equations]
    return "\n".join(equation_strings)


def create_arithmetic_dataset(output_dir, digits, size, test_fraction, char_level=True):
    equations = generate_equations(digits, size)
    if char_level:
        prepare_char_level_data(output_dir, equations, test_fraction)
    else:
        prepare_bpe_data(output_dir, equations, test_fraction)

def get_vocab_size(output_dir):
    meta_path = DATA_DIR / output_dir / 'meta.pkl'
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        return meta_vocab_size
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare a dataset for character-level or BPE language modeling.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the dataset files")
    parser.add_argument('--digits', type=int, required=True, help="Number of digits for each operand in the equations")
    parser.add_argument('--size', type=int, required=True, help="Number of equations to generate")
    parser.add_argument('--test_fraction', type=float, default=0.1, help="Fraction of data for validation set")
    parser.add_argument('--char_level', action='store_true', help="Use character-level encoding (default is BPE encoding)")

    args = parser.parse_args()
    
    create_arithmetic_dataset(
        output_dir=args.output_dir,
        digits=args.digits,
        size=args.size,
        test_fraction=args.test_fraction,
        char_level=args.char_level
    )
