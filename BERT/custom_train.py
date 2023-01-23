import time
import torch

from helpers import format_time

def custom_train(device, model, train_dataloader, optimizer, optimizer_scheduler, cross_entropy):
    t0 = time.time()
    total_loss = 0
    model.train()
  
    for step,batch in enumerate(train_dataloader):
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        preds = model(b_input_ids, b_input_mask)

        loss = cross_entropy(preds, b_labels)
        total_loss = total_loss + loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer_scheduler.step()

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

