import json

class DataReader():
    def __init__(self):
        with open('../Data/all_data.json', 'r', encoding='utf8') as f:
            self.data = json.loads(f.read())
    
    def get_all_target_paragraphs(self):
        return [x['targetParagraphs'] for x in self.data]

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
    
    def get_post_texts_and_first_n_words_as_strings(self, n):
        post_texts = self.get_all_post_texts()
        target_paragraphs_separated = self.get_all_target_paragraphs()

        merged_data = []
    
        for post_text, target_paragraphs in zip(post_texts, target_paragraphs_separated):
            post_words = post_text.split(' ')
            if post_words[-1][-1] not in '!?.':
                post_words[-1] += '.'

            if len(target_paragraphs) > 2:
                target_paragraphs = target_paragraphs[1:]

            target_paragraphs_words = []
            for paragraph in target_paragraphs:
                paragraph_words = paragraph.split(' ')
                target_paragraphs_words.extend(paragraph_words)
                if len(target_paragraphs_words) >= n + 1:
                    target_paragraphs_words = target_paragraphs_words[:n + 1]
                    break
            
            final_string = ' '.join(post_words) + ' ' + ' '.join(target_paragraphs_words)
            merged_data.append(final_string)

        return merged_data

class SplitDataReader():
    def __init__(self, folder_name):
        with open(f'./{folder_name}/train_input.json', 'r', encoding='utf8') as f:
            self.train_input = json.loads(f.read())
        with open(f'./{folder_name}/train_labels.json', 'r', encoding='utf8') as f:
            self.train_labels = json.loads(f.read())
        with open(f'./{folder_name}/val_input.json', 'r', encoding='utf8') as f:
            self.val_input = json.loads(f.read())
        with open(f'./{folder_name}/val_labels.json', 'r', encoding='utf8') as f:
            self.val_labels = json.loads(f.read())
        with open(f'./{folder_name}/test_input.json', 'r', encoding='utf8') as f:
            self.test_input = json.loads(f.read())
        with open(f'./{folder_name}/test_labels.json', 'r', encoding='utf8') as f:
            self.test_labels = json.loads(f.read())