import torch
import os

from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_readers import SplitDataReader
from my_bert import MyBERT
from train import train
from validate import validate

# pytorch memory management
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# training configuration
device = torch.device("cuda")
#model_name = "xlm-roberta-base"
#model_name = "bert-base-uncased"
#model_name = "microsoft/deberta-base"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = MyBERT(AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = 3,
    output_attentions = False,
    output_hidden_states = False
))
custom_model = True
lr = 2e-5
optimizer = AdamW(model.parameters(), lr = lr)
#folder_name = "post_texts"
#folder_name = "target_paragraphs"
folder_name = "post_texts_and_first_200_words"
split_data_reader = SplitDataReader(folder_name)
batch_size = 4
num_workers = 0
epochs = 4

# tokenize sentences
encoded_input_train = tokenizer(split_data_reader.train_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
encoded_input_val = tokenizer(split_data_reader.val_input, padding=True, truncation=True, max_length=512, return_tensors="pt")

# convert lists to tensors
train_inputs = encoded_input_train["input_ids"].clone().detach()
train_masks = encoded_input_train["attention_mask"].clone().detach()
train_labels = torch.tensor(split_data_reader.train_labels)

val_inputs = encoded_input_val["input_ids"].clone().detach()
val_masks = encoded_input_val["attention_mask"].clone().detach()
val_labels = torch.tensor(split_data_reader.val_labels)

# create data loaders
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)

validation_data = TensorDataset(val_inputs, val_masks, val_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, num_workers=num_workers)

model.cuda()
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, 
    num_warmup_steps = 0,
    num_training_steps = total_steps)

eval_acc = 0
for epoch in range(0, epochs):
    print(f"\n======== Epoch {epoch + 1} / {epochs} ========")
    
    avg_train_loss = train(device, model, train_dataloader, optimizer, optimizer_scheduler)

    print(f"\nAverage training loss: {avg_train_loss:.2f}")

    eval_acc = validate(device, model, validation_dataloader)
    print(f"\nAccuracy: {eval_acc:.2f}")

if custom_model:
    model_name = "my-" + model_name
model_path = f"./models/{folder_name}/{eval_acc:.2f}-{model_name.split('/')[-1]}-{epochs}e-{batch_size}bs-{lr}lr.pt"
torch.save(model.state_dict(), model_path)
print("Training complete!")