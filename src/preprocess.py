import os
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset

_CTRL = re.compile(r"[\u0000-\u001F\u007F]")

def _normalize_text(text: str) -> str:
    if text is None:
        return ""

    x = unicodedata.normalize("NFKC", text)
    x = _CTRL.sub(" ", x)
    x = x.replace("&nbsp;", " ").replace("&amp;", "&")
    x = re.sub(r"\r\n|\r", "\n", x)
    x = re.sub(r"[\.]{3,}", "…", x)
    x = re.sub(r"[!]{2,}", "!", x)
    x = re.sub(r"[?]{2,}", "?", x)
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()

def clean_dialogue_min(text: str) -> str:
    x = _normalize_text(text)
    x = re.sub(r"#\s*Person\s*1\s*#?", "[P1]", x, flags=re.IGNORECASE)
    x = re.sub(r"#\s*Person\s*2\s*#?", "[P2]", x, flags=re.IGNORECASE)
    x = re.sub(r"#\s*Person\s*3\s*#?", "[P3]", x, flags=re.IGNORECASE)

    x = re.sub(r"^\s*P1\s*:\s*", "[P1]: ", x, flags=re.IGNORECASE | re.MULTILINE)
    x = re.sub(r"^\s*P2\s*:\s*", "[P2]: ", x, flags=re.IGNORECASE | re.MULTILINE)
    x = re.sub(r"(\[P[123]\])\s*:\s*", r"\1: ", x)

    lines = [ln.strip() for ln in x.split("\n") if ln.strip()]
    return "\n".join(lines).strip()

def clean_summary_min(text: str) -> str:
    s = _normalize_text(text)
    s = re.sub(r"\s+", " ", s).strip()
    if not re.search(r"[\.!?…]$", s):
        s += "."
    return s

@dataclass
class LengthPolicy:
    max_words: int = 768

def truncate_dialogue_by_words(text: str, policy: LengthPolicy = LengthPolicy()) -> str:
    words = text.split()
    if len(words) <= policy.max_words:
        return text
    return " ".join(words[-policy.max_words:]).strip()

def filter_dataframe_min(df: pd.DataFrame, is_test: bool=False) -> pd.DataFrame:
    df = df.copy()
    if is_test:
        return df.reset_index(drop=True)

    df["dialogue_len"] = df["dialogue"].str.len()
    if "summary" in df.columns:
        df["summary_len"] = df["summary"].str.len()
        df = df[(df["summary_len"] > 10) & (df["dialogue_len"] < 2000)]
        df = df.drop(columns=["summary_len"])
    df = df.drop(columns=["dialogue_len"])
    return df.reset_index(drop=True)

def apply_preprocess(df: pd.DataFrame, is_test: bool=False, max_words: int=512) -> pd.DataFrame:
    df = df.copy()
    df["dialogue"] = df["dialogue"].apply(clean_dialogue_min)
    df["dialogue"] = df["dialogue"].apply(lambda x: truncate_dialogue_by_words(x, LengthPolicy(max_words)))
    if not is_test and "summary" in df.columns:
        df["summary"] = df["summary"].apply(clean_summary_min)
    df = filter_dataframe_min(df, is_test=is_test)
    return df.reset_index(drop=True)

def postprocess_summaries(summ_list):
    out = []
    for s in summ_list:
        if s is None:
            out.append("")
            continue
        s = re.sub(r"<\s*\/?\s*\w+\s*>", "", s)
        s = s.replace("[P1]", "").replace("[P2]", "").replace("[P3]", "")
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"([.]){2,}", ".", s)
        s = re.sub(r"(…){2,}", "…", s)
        if not re.search(r"[\.!?…]$", s):
            s += "."
        out.append(s)
    return out

class Preprocess:
    def __init__(self,
            bos_token: str,
            eos_token: str,
            ) -> None:
        
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train = True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[["fname", "dialogue", "summary"]]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[["fname", "dialogue"]]
            return test_df
        
    def make_input(self, dataset, is_test = False):
        if is_test:
            encoder_input = dataset["dialogue"]
            decoder_input = [self.bos_token] * len(dataset["dialogue"])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset["dialogue"]
            decoder_input = dataset["summary"].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset["summary"].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()
        
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2["decoder_input_ids"] = item2["input_ids"]
        item2["decoder_attention_mask"] = item2["attention_mask"]
        item2.pop("input_ids")
        item2.pop("attention_mask")
        item.update(item2)
        item["labels"] = self.labels[idx]
        return item
    
    def __len__(self):
        return self.len
    
class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return self.len

class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len
    
def prepare_train_dataset(config, preprocessor, tokenizer):
    train_path = os.path.join(config["general"]["data_path"], "train.csv")
    val_path = os.path.join(config["general"]["data_path"], "dev.csv")

    train_data = preprocessor.make_set_as_df(train_path)
    val_data = preprocessor.make_set_as_df(val_path)

    train_data = apply_preprocess(train_data, is_test=False, max_words=config["tokenizer"]["encoder_max_len"]).reset_index(drop=True)
    val_data = apply_preprocess(val_data, is_test=False, max_words=config["tokenizer"]["encoder_max_len"]).reset_index(drop=True)

    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)

    tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_outputs = tokenizer(decoder_output_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    labels_train = tokenized_decoder_outputs["input_ids"].clone()
    labels_train[labels_train == tokenizer.pad_token_id] = -100

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, labels_train, len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_outputs = tokenizer(decoder_output_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    labels_val = val_tokenized_decoder_outputs["input_ids"].clone()
    labels_val[labels_val == tokenizer.pad_token_id] = -100

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, labels_val, len(encoder_input_val))

    return train_inputs_dataset, val_inputs_dataset

def prepare_test_dataset(config, preprocessor, tokenizer):
    test_path = os.path.join(config["general"]["data_path"], "test.csv")
    test_data = preprocessor.make_set_as_df(test_path, is_train=False)
    test_data = apply_preprocess(test_data, is_test=True, max_words=config["tokenizer"]["encoder_max_len"])
    test_data = test_data.reset_index(drop=True)
    test_id = test_data["fname"].tolist()

    encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
    test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))

    return test_data, test_encoder_inputs_dataset