"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import argparse
from omegaconf import OmegaConf
from pathlib import Path

from .model import GPTConfig, GPT
from .utils import MODELS_DIR, CONFIG_DIR, DATA_DIR

def get_model(conf):
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
 
   
    # model
    if conf.init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = MODELS_DIR / conf.out_dir / 'ckpt.pt'
        checkpoint = torch.load(ckpt_path, map_location=conf.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif conf.init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(conf.init_from, dict(dropout=0.0))

    model.eval()
    model.to(conf.device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if conf.init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = DATA_DIR / checkpoint['config']['dataset'] / 'meta.pkl'
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return model, encode, decode

def generate_stuff(conf, silent=False):
    model, encode, decode = get_model(conf)
    device_type = 'cuda' if 'cuda' in conf.device else 'cpu' # for later use in torch.autocast
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # encode the beginning of the prompt
    if conf.start.startswith('FILE:'):
        with open(conf.start[5:], 'r', encoding='utf-8') as f:
            conf.start = f.read()
    start_ids = encode(conf.start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=conf.device)[None, ...])

    # run generation
    results = []
    with torch.no_grad():
        with ctx:
            for k in range(conf.num_samples):
                y = model.generate(x, conf.max_new_tokens, temperature=conf.temperature, top_k=conf.top_k)
                generated = decode(y[0].tolist())
                results.append(generated)
                if not silent:
                    print(generated)
                    print('---------------')
    return results

def generate_many(conf, prompts):
    model, encode, decode = get_model(conf)
    device_type = 'cuda' if 'cuda' in conf.device else 'cpu' # for later use in torch.autocast
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # run generation
    results = []
    with torch.no_grad():
        with ctx:
            for prompt in prompts:
                start_ids = encode(prompt)
                x = (torch.tensor(start_ids, dtype=torch.long, device=conf.device)[None, ...])

                y = model.generate(x, conf.max_new_tokens, 
                temperature=conf.temperature, top_k=conf.top_k)
                generated = decode(y[0].tolist())
                results.append(generated)

    return results

def make_sampling_config(config_name=None, overrides=None):
    base_cfg = OmegaConf.load(CONFIG_DIR / "base_sample.yaml")

    if config_name:
        override_cfg = OmegaConf.load(CONFIG_DIR / config_name)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg

    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configurations with OmegaConf")
    parser.add_argument("--config_file", type=str, help="Path to an additional config file to override the base config")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Additional parameters to override as KEY=VALUE")

    args = parser.parse_args()

    cfg = make_sampling_config(args.config_file, OmegaConf.from_dotlist(args.overrides))

    # Now `cfg` holds the final configuration with all overrides applied
    print(OmegaConf.to_yaml(cfg))  # Print the final config for demonstration
    generate_stuff(cfg)
