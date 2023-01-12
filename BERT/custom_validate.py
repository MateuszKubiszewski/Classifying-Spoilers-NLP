import torch

from helpers import flat_accuracy

def custom_validate(device, model, validation_dataloader):
    model.eval()

    eval_steps = 0
    eval_accuracy = 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            preds = model(b_input_ids, b_input_mask)

    preds = preds.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    batch_accuracy = flat_accuracy(preds, label_ids)
    
    eval_accuracy += batch_accuracy
    eval_steps += 1

    return eval_accuracy/eval_steps