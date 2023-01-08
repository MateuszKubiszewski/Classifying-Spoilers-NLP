import json

class DataReader():
    def __init__(self):
        with open('../Data/all_data.json', 'r', encoding='utf8') as f:
            self.data = json.loads(f.read())

    def get_all_target_paragraphs_as_strings(self):
        return [' '.join(x['targetParagraphs']) for x in self.data]

    def get_all_labels(self):
        return [x['tags'][0] for x in self.data]
    
    def get_all_labels_as_integers(self, phrase=0, passage=1, multi=2):
        labelled_data_ints = []
        for x in self.get_all_labels():
            if x == 'phrase':
                labelled_data_ints.append(phrase)
            if x == 'passage':
                labelled_data_ints.append(passage)
            if x == 'multi':
                labelled_data_ints.append(multi)
        return labelled_data_ints
    
    def get_all_post_texts(self):
        return [x['postText'][0] for x in self.data]


class SplitDataReader():
    def __init__(self, folder_name):
        with open(f'./{folder_name}/{folder_name}_train_input.json', 'r', encoding='utf8') as f:
            self.train_input = json.loads(f.read())
        with open(f'./{folder_name}/{folder_name}_train_labels.json', 'r', encoding='utf8') as f:
            self.train_labels = json.loads(f.read())
        with open(f'./{folder_name}/{folder_name}_val_input.json', 'r', encoding='utf8') as f:
            self.val_input = json.loads(f.read())
        with open(f'./{folder_name}/{folder_name}_val_labels.json', 'r', encoding='utf8') as f:
            self.val_labels = json.loads(f.read())
        with open(f'./{folder_name}/{folder_name}_test_input.json', 'r', encoding='utf8') as f:
            self.test_input = json.loads(f.read())
        with open(f'./{folder_name}/{folder_name}_test_labels.json', 'r', encoding='utf8') as f:
            self.test_labels = json.loads(f.read())