import os
from typing import Union
from rouge import Rouge

import wandb
import numpy as np
from transformers import EvalPrediction, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, BartConfig, AutoTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers.trainer_utils import IntervalStrategy

from utils import DEFAULTS

def make_compute_metrics(tokenizer):
    def _compute_metrics(eval_pred: Union[EvalPrediction, tuple]):
        if isinstance(eval_pred, EvalPrediction):
            preds = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.array(preds)
        labels = np.array(labels)

        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        pad_id = tokenizer.pad_token_id or 0
        preds = np.where(preds == -100, pad_id, preds)
        labels = np.where(labels == -100, pad_id, labels)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        correct = 0; total = 0
        for p, l in zip(decoded_preds, decoded_labels):
            correct += int(p.strip() == l.strip())
            total += 1
        acc = correct / max(total, 1)
        return {"simple_exact_match": acc}
    return _compute_metrics

def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    cfg_gen = config["general"]
    cfg_trn = {**DEFAULTS, **config.get("training", {})}

    strategy = IntervalStrategy("steps")

    bf16_check = config["training"]["bf16"],
    fp16_check = config["training"]["fp16"],
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg_gen["output_dir"],
        overwrite_output_dir=cfg_trn["overwrite_output_dir"],
        num_train_epochs=float(cfg_trn["num_train_epochs"]),
        learning_rate=float(cfg_trn["learning_rate"]),
        per_device_train_batch_size=int(cfg_trn["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg_trn["per_device_eval_batch_size"]),
        warmup_ratio=float(cfg_trn["warmup_ratio"]),
        weight_decay=float(cfg_trn["weight_decay"]),
        save_total_limit=int(cfg_trn["save_total_limit"]),
        logging_steps=float(cfg_trn["logging_steps"]),
        save_steps=float(cfg_trn["save_steps"]),
        eval_steps=float(cfg_trn["eval_steps"]),
        report_to=["wandb"],

        eval_strategy=strategy,
        logging_strategy=strategy,
        save_strategy=strategy,

        bf16=bool(bf16_check),
        fp16=bool(not bf16_check and fp16_check),
        predict_with_generate=config["training"]["predict_with_generate"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    wandb.init(
        entity=config["wandb"]["entity"],
        name=config["wandb"]["name"],
        project=config["wandb"]["project"],
    )
    os.environ["WANDB_LOG_MODEL"]="end"
    os.environ["WANDB_WATCH"]="false"

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config["training"]["early_stopping_patience"],
        early_stopping_threshold=config["training"]["early_stopping_threshold"],
    )

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[MyCallback]
    )

    return trainer

def load_tokenizer_and_model_for_train(config, device):
    model_name = config["general"]["model_name"]
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config["general"]["model_name"], config=bart_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if generate_model.config.pad_token_id is None:
        generate_model.config.pad_token_id = tokenizer.pad_token_id

    special_tokens_dict = {"additional_special_tokens": config["tokenizer"]["special_tokens"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)

    return generate_model, tokenizer

def load_tokenizer_and_model_for_test(config, ckpt_path, device):
    model_name = config["general"]["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {"additional_special_tokens": config["tokenizer"]["special_tokens"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    try:
        model = BartForConditionalGeneration.from_pretrained(ckpt_path)
        print(model)
        # ckpt_path:  ./outputs/checkpoints/digit82/kobart-summarization/checkpoint-500
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path)
    

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    return model, tokenizer