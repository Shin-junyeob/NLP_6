import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config_info
from preprocess import Preprocess, prepare_test_dataset
from model import load_tokenizer_and_model_for_test
from train import train

def main(config):
    ckpt_path = train(config)
    print("ckpt_path: ", ckpt_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, ckpt_path, device)
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config["inference"]["batch_size"])

    summary = []; text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item["ID"])
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to(device),
                attention_mask=item["attention_mask"].to(device),
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[tokenizer.pad_token_id]],
            )
            for ids in generated_ids:
                result = tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
                )
                summary.append(result)

    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token, " ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output

if __name__ == "__main__":
    config = config_info()
    main(config)