#!/usr/bin/env python3
import argparse
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def run_baseline(input_file, output_file):
    input_df = pd.read_json(path_or_buf=input_file, lines=True)
    uuids = input_df["uuid"]
    input_df = data_processing(input_df)
    input_tokens = tokenizing(input_df)

    input_seq = input_tokens['input_ids'].clone().detach()
    input_mask = input_tokens['attention_mask'].clone().detach()

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained("/model").to(device)
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
    ret = []
    for _, i in dataframe.iterrows():
        post_words = i['postText'][0].split(' ')
        if post_words[-1][-1] not in '!?.':
            post_words[-1] += '.'

        target_paragraphs = i['targetParagraphs']
        if len(target_paragraphs) > 2:
            target_paragraphs = target_paragraphs[1:]

        target_paragraphs_words = []
        for paragraph in target_paragraphs:
            paragraph_words = paragraph.split(' ')
            target_paragraphs_words.extend(paragraph_words)
            if len(target_paragraphs_words) >= 201:
                target_paragraphs_words = target_paragraphs_words[:201]
                break

        final_string = ' '.join(post_words) + ' ' + ' '.join(target_paragraphs_words)
        ret += [{'text': final_string, 'uuid': i['uuid']}]

    return pd.DataFrame(ret)


def tokenizing(dataframe):
    tokenizer = AutoTokenizer.from_pretrained("/tokenizer")
    encoded_input = tokenizer(dataframe.text.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

    return encoded_input


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)