from my_bert import MyBERT
from transformers import AutoModelForSequenceClassification

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = 3,
    output_attentions = False,
    output_hidden_states = False
)

model2 = MyBERT(model)
print(model2)