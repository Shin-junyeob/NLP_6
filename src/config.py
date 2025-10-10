import yaml

from transformers import AutoTokenizer

def config_info():
    config_path = "./config/config.yaml"
    def _config_info():
        MODEL_NAME = "digit82/kobart-summarization"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

        config_data = {
            "general": {
                "data_path": "./data/",
                "model_name": MODEL_NAME,
                "output_dir": f"./outputs/checkpoints/{MODEL_NAME}/"
            },
            "tokenizer": {
                "encoder_max_len": 512,
                "decoder_max_len": 100,
                "bos_token": f"{tokenizer.bos_token}",
                "eos_token": f"{tokenizer.eos_token}",
                "special_tokens": ["#Person1#", "#Person2#", "#Person3#", "#PhoneNumber#", "#Address#", "#PassportNumber#"]
            },
            "training": {
                "overwritre_output_dir": True,
                "num_train_epochs": 20,
                "learning_rate": 1e-5,
                "per_device_train_batch_size": 50,
                "per_device_eval_batch_size": 32,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "lr_scheduler_type": 'cosine',
                "optim": 'adamw_torch',
                "gradient_accumulation_steps": 1,
                "evaluation_strategy": 'epoch',
                "save_strategy": 'epoch',
                "save_total_limit": 5,
                "bf16": True,
                "fp16": False,
                "load_best_model_at_end": True,
                "seed": 42,
                "logging_dir": "./logs",
                "logging_strategy": "epoch",
                "predict_with_generate": True,
                "generation_max_length": 100,
                "do_train": True,
                "do_eval": True,
                "early_stopping_patience": 3,
                "early_stopping_threshold": 0.001,
                "report_to": ["wandb"]
            },
            "wandb": {
                "entity": "junyub029-github",
                "project": "nlp_project",
                "name": "baseline_code"
            },
            "inference": {
                "ckpt_path": f"./outputs/checkpoints/{MODEL_NAME}",
                "result_path": f"./outputs/prediction/{MODEL_NAME}/",
                "no_repeat_ngram_size": 2,
                "early_stopping": True,
                "generate_max_length": 100,
                "num_beams": 4,
                "batch_size": 32,
                "remove_tokens": ["<usr>", f"{tokenizer.bos_token}", f"{tokenizer.eos_token}", f"{tokenizer.pad_token}"]
            }
        }

        config_path = "./config/config.yaml"
        with open(config_path, "w") as file:
            yaml.dump(config_data, file, allow_unicode=True)

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        return config
    return _config_info()
