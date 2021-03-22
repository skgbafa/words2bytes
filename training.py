# imports
import time
import math
import torch
import wandb

from constants import *
from utils import extract_config
from data import get_batch


# training loop
def train(model, batches, config, runtime, epoch, artifacts):
    max_seq_len = extract_config(config, "max_seq_len")

    # get runtime vars
    criterion = runtime["criterion"]
    optimizer = runtime["optimizer"]
    scheduler = runtime["scheduler"]
    ntokens = runtime["ntokens"]
    device = runtime["device"]

    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(max_seq_len).to(device)
    for batch, i in enumerate(range(0, batches.size(0) - 1, max_seq_len)):
        data, targets = get_batch(max_seq_len, batches, i)
        optimizer.zero_grad()
        if data.size(0) != max_seq_len:
            src_mask = model.generate_square_subsequent_mask(
                data.size(0)).to(device)
        print("data.shape", data.shape)
        print("targets.shape", targets.shape)
        print("src_mask.shape", src_mask.shape)

        output = model(data, src_mask)
        print("output", output.shape)

        output_flat = output.view(-1, ntokens)

        print("output_flat", output_flat.shape)
        loss = criterion(output_flat, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "batch_loss": loss.item(),
            "ppl": math.exp(loss.item()),
            "learning_rate": scheduler.get_lr()[0],
        })

        total_loss += loss.item()
        log_interval = 200
        cur_loss = total_loss / log_interval
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(
                          batches) // max_seq_len, scheduler.get_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# evaluation


def evaluate(model, data_source, config, runtime):
    max_seq_len = extract_config(config, "max_seq_len")

    # get runtime vars
    criterion = runtime["criterion"]
    ntokens = runtime["ntokens"]
    device = runtime["device"]

    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(max_seq_len).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, max_seq_len):
            data, targets = get_batch(max_seq_len, data_source, i)

            if data.size(0) != max_seq_len:
                src_mask = model.generate_square_subsequent_mask(
                    data.size(0)).to(device)

            output = model(data, src_mask)

            output_flat = output.view(-1, ntokens)
            print("output_flat", output_flat.shape)
            loss = criterion(output_flat, targets)
            total_loss += len(data) * loss.item()

            wandb.log({
                "batch_loss": loss.item(),
                "ppl": math.exp(loss.item()),
            })
    return total_loss / (len(data_source) - 1)
