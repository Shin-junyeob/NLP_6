import os

import pandas as pd
from torch.utils.data import Dataset

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
    test_id = test_data["fname"]

    encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
    test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))

    return test_data, test_encoder_inputs_dataset