import json

from data_reader import DataReader
from sklearn.model_selection import train_test_split

def split_data(input_data, labels):
    train_input, temp_input, train_labels, temp_labels = train_test_split(input_data, labels, 
        random_state=2023, test_size=0.2, stratify=labels)

    val_input, test_input, val_labels, test_labels = train_test_split(temp_input, temp_labels,
        random_state=2023, test_size=0.5, stratify=temp_labels)

    return train_input, train_labels, val_input, val_labels, test_input, test_labels

def save_split_data_into_json(folder_name, train_input, train_labels, val_input, val_labels, test_input, test_labels):
    with open(f'./{folder_name}/{folder_name}_train_input.json', 'w+', encoding="utf8") as dest:
        dest.write(json.dumps(train_input))
    with open(f'./{folder_name}/{folder_name}_train_labels.json', 'w+', encoding="utf8") as dest:
        dest.write(json.dumps(train_labels))
    with open(f'./{folder_name}/{folder_name}_val_input.json', 'w+', encoding="utf8") as dest:
        dest.write(json.dumps(val_input))
    with open(f'./{folder_name}/{folder_name}_val_labels.json', 'w+', encoding="utf8") as dest:
        dest.write(json.dumps(val_labels))
    with open(f'./{folder_name}/{folder_name}_test_input.json', 'w+', encoding="utf8") as dest:
        dest.write(json.dumps(test_input))
    with open(f'./{folder_name}/{folder_name}_test_labels.json', 'w+', encoding="utf8") as dest:
        dest.write(json.dumps(test_labels))

if __name__ == "__main__":
    data_reader = DataReader()
    labels = data_reader.get_all_labels_as_integers()
    post_texts = data_reader.get_all_post_texts()
    target_paragraphs = data_reader.get_all_target_paragraphs_as_strings()

    train_input, train_labels, val_input, val_labels, test_input, test_labels = split_data(post_texts, labels)
    save_split_data_into_json('post_text', train_input, train_labels, val_input, val_labels, test_input, test_labels)

    train_input, train_labels, val_input, val_labels, test_input, test_labels = split_data(target_paragraphs, labels)
    save_split_data_into_json('target_paragraphs', train_input, train_labels, val_input, val_labels, test_input, test_labels)