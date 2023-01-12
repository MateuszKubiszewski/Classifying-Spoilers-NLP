import torch

from helpers import flat_accuracy

def validate(device, model, validation_dataloader):
    model.eval()

    eval_steps = 0
    eval_accuracy = 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        batch_accuracy = flat_accuracy(logits, label_ids)
        
        eval_accuracy += batch_accuracy
        eval_steps += 1

    return eval_accuracy/eval_steps