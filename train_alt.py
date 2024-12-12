"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import json
from tqdm import tqdm
from contextlib import nullcontext
from omegaconf import OmegaConf
import argparse


import numpy as np
import torch
from torch._prims_common import check
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from .model import GPTConfig, GPT
from .utils import BASE_PATH, DATA_DIR, MODELS_DIR, logger
from .data_prep import load_encoder

def init_model(model_cfg):
    return GPT(model_cfg)

def load_model_history(model_name):
    ckpt_path = MODELS_DIR / model_name / 'ckpt.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    return checkpoint['history']

def load_model(model_name, device):
    ckpt_path = MODELS_DIR / model_name / 'ckpt.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    model = GPT(OmegaConf.create(checkpoint['model_args']))
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num_start = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # print(f'best val loss = {best_val_loss}')
    model.load_encoder(checkpoint['encoder'])
    model.to(device)
    return model


def train(model, dataset, train_cfg, env_cfg):
    
    model.encoder = load_encoder(dataset)
    context_size = model.config.block_size
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=env_cfg.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(env_cfg.device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert train_cfg.gradient_accumulation_steps % ddp_world_size == 0
        train_cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = train_cfg.gradient_accumulation_steps * ddp_world_size * train_cfg.batch_size * context_size
    logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")
    model_dir = MODELS_DIR / train_cfg.out_dir
    if master_process:
        os.makedirs(model_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in env_cfg.device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    dataset_dir = DATA_DIR / dataset

    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(dataset_dir / 'train.bin', dtype=np.uint16, mode='r')
        else:
            data = np.memmap(dataset_dir / 'val.bin', dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - context_size, (train_cfg.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+context_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+context_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(env_cfg.device, non_blocking=True), y.pin_memory().to(env_cfg.device, non_blocking=True)
        else:
            x, y = x.to(env_cfg.device), y.to(env_cfg.device)
        return x, y

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num_start = 0
    best_val_loss = 1e9

    model.to(env_cfg.device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(train_cfg.weight_decay, train_cfg.learning_rate, (train_cfg.beta1, train_cfg.beta2), device_type)
    if train_cfg.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        logger.info("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(train_cfg.eval_iters)
            for k in range(train_cfg.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = float(losses.mean())
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < train_cfg.warmup_iters:
            return train_cfg.learning_rate * it / train_cfg.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > train_cfg.lr_decay_iters:
            return train_cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - train_cfg.warmup_iters) / (train_cfg.lr_decay_iters - train_cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return train_cfg.min_lr + coeff * (train_cfg.learning_rate - train_cfg.min_lr)

    # logging
    if train_cfg.wandb_log and master_process:
        import wandb
        wandb.init(project=train_cfg.wandb_project, name=train_cfg.wandb_run_name, config=train_cfg)

    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    history = []
    for iter_num in tqdm(range(iter_num_start, train_cfg.max_iters)):

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if train_cfg.decay_lr else train_cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % train_cfg.eval_interval == 0 and master_process:
            losses = estimate_loss()
            logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            stats = {
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                }
            history.append(stats)
            if train_cfg.wandb_log:
                wandb.log(stats)
            if losses['val'] < best_val_loss or train_cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'encoder': raw_model.save_encoder(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': raw_model.config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': train_cfg,
                        'history': history
                    }
                    logger.info(f"saving checkpoint to {train_cfg.out_dir}")
                    torch.save(checkpoint, model_dir / 'ckpt.pt')
        if iter_num == 0 and train_cfg.eval_only:
            logger.info('BREAKING BECAUSE EVAL ONLY')
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(train_cfg.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == train_cfg.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / train_cfg.gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if train_cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % train_cfg.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * train_cfg.gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(train_cfg.batch_size * train_cfg.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        local_iter_num += 1

    with open(model_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load configurations with OmegaConf")
    parser.add_argument("--config_file", type=str, help="Path to an additional config file to override the base config")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Additional parameters to override as KEY=VALUE")

    args = parser.parse_args()

    cfg = make_train_config(args.config_file, OmegaConf.from_dotlist(args.overrides))
    # Now `cfg` holds the final configuration with all overrides applied
    print(OmegaConf.to_yaml(cfg))  # Print the final config for demonstration
    train(cfg)
