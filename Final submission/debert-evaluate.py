#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='DeBERT model which predicts spoiler type based on clickbait content and first 200 words of the article.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def load_input(df):
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)
    
    ret = []
    for _, i in df.iterrows():
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


def use_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def get_device():
    return torch.device("cuda" if use_cuda() else "cpu")


def predict(df):
    df = load_input(df)
    #labels = ['phrase', 'passage', 'multi']
    
    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained('./debert_post_text_200_words', num_labels = 3)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    uuids = list(df['uuid'])
    texts = list(df['text'])
    
    encoded_input_test = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    test_inputs = encoded_input_test['input_ids'].clone().detach()
    test_masks = encoded_input_test['attention_mask'].clone().detach()
    
    for uuid, test_input, test_mask in zip(uuids, test_inputs, test_masks):
        print(test_input)
        print(test_mask)
        with torch.no_grad():
            outputs = model(test_input.to(device), test_mask.to(device))
        print(outputs)
        break

    #logits = outputs[0]
    #logits = logits.detach().cpu().numpy()
    #label_ids = b_labels.to('cpu').numpy()
    #for i in range(len(df)):
        #yield {'uuid': uuids[i], 'spoilerType': labels[np.argmax(predictions[i])]}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    #args = parse_args()
    #run_baseline(args.input, args.output)
    run_baseline('./data/validation.jsonl', './output.json')