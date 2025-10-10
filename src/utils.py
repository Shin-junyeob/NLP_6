import random
import os

import numpy as np
import torch
import pytorch_lightning as pl
from transformers import set_seed
from dotenv import load_dotenv, find_dotenv

import wandb

DEFAULTS = {
    "evaluation_strategy": "steps",   # "steps" | "epoch" | "no"
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
    "num_train_epochs": 1.0,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "save_total_limit": 2,
    "overwrite_output_dir": True,
}

def _settings(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    set_seed(seed)
    pl.seed_everything(seed, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)

    os.environ["WANDB_LOG_MODEL"] = "end"
    wandb.login(key=os.getenv("WANDB_API_KEY"))