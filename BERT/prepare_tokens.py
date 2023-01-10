from transformers import BertTokenizerFast, BertTokenizer

from data_readers import DataReader

def test():
    merged_data = []
    biggest_sentence = ""
    max_len = 0
    for x, y in zip(post_texts, target_paragraphs_separated):
        if len(y) == 0:
            print(y)
            raise Exception("what the fuck")
        words = y[0].split(' ')
        if len(words) > max_len:
            max_len = len(words)
            biggest_sentence = y[0]
        if len(y) == 1:
            #print(y)
            #raise Exception("what the fuck")
            merged_data.append(' '.join([x, y[0]]))
        #if len(y) == 2:
            #merged_data.append(' '.join([x, y[0], y[1]]))
            #merged_data.append(' '.join([x, y[0]]))
        #if len(y) > 2:
            #merged_data.append(' '.join([x, y[1], y[2]]))
            #merged_data.append(' '.join([x, y[1]]))
    #print(merged_data[0])
    #print(biggest_sentence)
    return merged_data

def prepare_merged_data(post_texts, target_paragraphs_separated):
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
            if len(target_paragraphs_words) >= 201:
                target_paragraphs_words = target_paragraphs_words[:201]
                break
        
        final_string = ' '.join(post_words) + ' ' + ' '.join(target_paragraphs_words)
        merged_data.append(final_string)

    return merged_data

if __name__ == "__main__":
    data_reader = DataReader()
    post_texts = data_reader.get_all_post_texts()
    target_paragraphs = data_reader.get_all_target_paragraphs_as_strings()
    target_paragraphs_separated = data_reader.get_all_target_paragraphs()

    #tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

    merged_data = prepare_merged_data(post_texts, target_paragraphs_separated)

    encoded_input = tokenizer(merged_data, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoded_input['input_ids']
    token_type_ids = encoded_input['token_type_ids']
    attention_mask = encoded_input['attention_mask']

    print('Max sentence length: ', max([len(sen) for sen in input_ids]))
    print('Max sentence: ', max(merged_data, key=len))        