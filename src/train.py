import os

import torch
import pandas as pd
import wandb

from utils import DEFAULTS, _settings
from config import config_info
from preprocess import Preprocess, prepare_train_dataset
from model import load_tokenizer_and_model_for_train, load_trainer_for_train

def train(config):
    _settings()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    preprocessor = Preprocess(config["tokenizer"]["bos_token"], config["tokenizer"]["eos_token"])

    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, tokenizer)

    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    trainer.train()

    wandb.finish()
    return trainer.state.best_model_checkpoint