import argparse

import pandas as pd
import evaluate
import matplotlib.pyplot as plt

rouge = evaluate.load("rouge")

def rouge_1_sample(row):
    preds = row["summary"]
    refs = row["dialogue"].split("\n")[0]
    res = rouge.compute(
        predictions=[preds],
        references=[refs],
        rouge_types=["rougeL"]
    )
    score = res["rougeL"]
    try:
        return float(score.mid.fmeasure)
    except AttributeError:
        return float(score)

def simple_eda(is_train: str = 'train'):
    path = f'../data/{is_train}.csv'
    target = pd.read_csv(path)

    target['dialogue_len'] = target['dialogue'].str.len()
    target['summary_len'] = target['summary'].str.len()

    print(f'#### {is_train}.csv 분석 ####\n')
    res = target[['dialogue_len', 'summary_len']].describe()
    print(res)
    print('\n#### topic별 분석 ####\n')
    res = target.groupby('topic')[['dialogue_len', 'summary_len']].mean().sort_values('dialogue_len')
    print(res)

    target['rouge_sample'] = target.sample(200).apply(rouge_1_sample, axis=1)
    print('\n#### 요약 품질 EDA ####\n')
    res = target['rouge_sample'].mean()
    print(res)

    # print('\n#### 시각화 ####\n')
    # plt.hist(target['dialogue_len'], bins=50, alpha=0.7, label='dialogue')
    # plt.hist(target['summary_len'], bins=50, alpha=0.7, label='summary')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" Run simple EDA for dialogue summarization data.")
    parser.add_argument("dataset", type=str, choices=["train", "dev"], help="Which dataset to analyze: train or dev")
    args = parser.parse_args()

    simple_eda(args.dataset)