#!/usr/bin/env python3
import argparse
import json
import pandas as pd
from transformers import AutoTokenizer, DebertaForSequenceClassification
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def run_baseline(input_file, output_file):
    input_df = pd.read_json(path_or_buf=input_file, lines = True)
    uuids = input_df["uuid"]
    input_df = data_processing(input_df)

    input_tokens = tokenizing(input_df)

    input_seq = torch.tensor(input_tokens['input_ids'])
    input_mask = torch.tensor(input_tokens['attention_mask'])

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    model = DebertaForSequenceClassification.from_pretrained("/model").to(device)
    labels = {0: "multi", 1: "passage", 2: "phrase"}

    with open(output_file, 'w') as out:
        model.eval()
        with torch.no_grad():
            for i in range(len(input_seq)):
                preds = model(input_seq[i].unsqueeze(dim=0).to(device), input_mask[i].unsqueeze(dim=0).to(device))
                preds = np.argmax(preds.logits.cpu(), axis=1).tolist()[0]
                preds = labels[preds]
                prediction = {'uuid': uuids[i], 'spoilerType': preds}
                out.write(json.dumps(prediction) + '\n')


def data_processing(dataframe):
    columns_to_drop = ["uuid", "postId", "postPlatform", "targetDescription",
                       "targetKeywords", "targetMedia", "targetUrl"]
    input_df = dataframe.drop(columns=columns_to_drop)

    columns_with_brackets = ["postText", "targetParagraphs"]
    input_df = _delete_brackets(input_df, columns_with_brackets)

    return input_df


def tokenizing(dataframe):
    tokenizer = AutoTokenizer.from_pretrained("/tokenizer")
    #FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/huggingface/hub/models--tokenizer.json/refs/main'
    #huggingface_hub.utils._validators.HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: './tokenizer.json'.
    #   f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
    #OSError: /tokenizer does not appear to have a file named config.json. Checkout 'https://huggingface.co//tokenizer/None' for available files.
    tokens_post = tokenizer.batch_encode_plus(
        dataframe.postText.tolist(),
        max_length=28,
        pad_to_max_length=True,
        truncation=True
    )
    tokens_title = tokenizer.batch_encode_plus(
        dataframe.targetTitle.tolist(),
        max_length=33,
        pad_to_max_length=True,
        truncation=True
    )

    tokens_par = tokenizer.batch_encode_plus(
        dataframe.targetParagraphs.tolist(),
        max_length=512 - 28 - 33,
        pad_to_max_length=True,
        truncation=True
    )

    input_tokens = _prepare_tokens(tokens_post, tokens_title, tokens_par)
    return input_tokens


def _prepare_tokens(post, title, paragraphs):
    result = {'input_ids': [], 'attention_mask': []}
    for i in range(len(post["input_ids"])):
        post_data = post['input_ids'][i]
        post_mask = post['attention_mask'][i]
        title_data = title['input_ids'][i]
        title_mask = title['attention_mask'][i]
        paragraphs_data = paragraphs['input_ids'][i]
        paragraphs_mask = paragraphs['attention_mask'][i]

        new_data = post_data + title_data + paragraphs_data
        new_mask = post_mask + title_mask + paragraphs_mask

        result['input_ids'].append(new_data)
        result['attention_mask'].append(new_mask)
    return result


def _delete_brackets(dataframe, columns_with_brackets):
    for index, data_row in dataframe.iterrows():
        for column_name in columns_with_brackets:
            data_row[column_name] = data_row[column_name][0]
    return dataframe


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)