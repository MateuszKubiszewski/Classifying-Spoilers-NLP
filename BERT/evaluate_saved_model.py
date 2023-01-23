import time
import torch

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_readers import SplitDataReader
from helpers import flat_accuracy, format_time

# pytorch memory management
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:256'

# specify GPU
device = torch.device("cuda")

saved_weights_path = 'saved_weights.pt'
data_folder_name = 'target_paragraphs'
batch_size = 4
num_workers = 0

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 3,  
    output_attentions = False,
    output_hidden_states = False
)
model.load_state_dict(torch.load(saved_weights_path))
model.cuda()

split_data_reader = SplitDataReader(data_folder_name)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

encoded_input_test = tokenizer(split_data_reader.test_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
test_inputs = encoded_input_test['input_ids'].clone().detach()
test_masks = encoded_input_test['attention_mask'].clone().detach()
test_labels = torch.tensor(split_data_reader.test_labels)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)

print('Predicting labels for {:,} test sentences...'.format(len(test_inputs)))

model.eval()
predictions, true_labels = [], []
t0 = time.time()
eval_steps = 0
eval_accuracy = 0

for (step, batch) in enumerate(test_dataloader):
    batch = tuple(t.to(device) for t in batch)

    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    b_input_ids, b_input_mask, b_labels = batch
  
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    eval_accuracy += flat_accuracy(logits, label_ids)
    eval_steps += 1

print("Accuracy: {0:.2f}".format(eval_accuracy/eval_steps))