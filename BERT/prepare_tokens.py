from transformers import BertTokenizerFast, BertTokenizer

from data_readers import DataReader

def get_post_text_tokens():
    data_reader = DataReader()
    post_texts = data_reader.get_all_post_texts()

if __name__ == "__main__":
    data_reader = DataReader()
    post_texts = data_reader.get_all_post_texts()
    target_paragraphs = data_reader.get_all_target_paragraphs_as_strings()

    #tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    encoded_input = tokenizer(target_paragraphs, padding=True, truncation=True, max_length=4096, return_tensors="pt")
    input_ids = encoded_input['input_ids']
    token_type_ids = encoded_input['token_type_ids']
    attention_mask = encoded_input['attention_mask']

    print('Max sentence length: ', max([len(sen) for sen in input_ids]))