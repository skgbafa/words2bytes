
# imports
from typing import Optional, Any
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder, LayerNorm

import enum
import io
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchtext import datasets, vocab
from torchtext.data import Field, BPTTIterator
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from transformers import AutoTokenizer

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np 

import wandb

from transformer import DecoderOnlyTransformer
from constants import *




# experiment generation
def generateExperiements():
    WANDB_ENTITY = "skgbafa"
    # WANDB_ENTITY = "openai-scholars"
    WANDB_PROJECT = ""

    # experiment_datasets = [ Dataset.PennTreebank.name, Dataset.WikiText2.name, Dataset.WikiText103.name ]
    experiment_datasets = [Dataset.WikiText2.name]
    experiment_segmentation = [
        Segmentation.Word.name, Segmentation.Character.name]
    # for each dataset
    #
    sweep_parameters = {
        "n_attention_heads": {
            "values": [2, 3, ]
        },
        "n_decoder_layers": {
            "values": [2, 4, 6]
        },
        "dataset": {
            "values": experiment_datasets
        },
        "n_epochs": {
            "values": [3]
        },
        "segmentation": {
            "values": experiment_segmentation
        }
    }

    sweep_config = {
        "name": "Experamental Sweeps",
        "method": "grid",
        "parameters": sweep_parameters
    }

    sweep_id = wandb.sweep(sweep_config, entity=WANDB_ENTITY)

    return sweep_id




# training loop
def train(model, config, epoch, artifacts):
    max_seq_len = extract_config(config, "max_seq_len")
    
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(max_seq_len).to(device)
    for batch, i in enumerate(range(0, train_data_batches.size(0) - 1, max_seq_len)):
        data, targets = get_batch(max_seq_len, train_data_batches, i)
        optimizer.zero_grad()
        if data.size(0) != max_seq_len:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        # print(data.dtype)
        # output = model(data, targets)
        reshape_seq_len = min(data.size(0), max_seq_len)
        targets_flat = targets.reshape(reshape_seq_len, targets.size(0)//reshape_seq_len)
        output = model(data, src_mask)
        # output = model(data, targets_flat, src_mask)
        # output = model(data, targets_flat, src_mask, src_mask)

        output.view(-1, ntokens)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # update_artifact_loss(artifacts, 'training', 'CrossEntropyLoss', epoch, batch, loss.item())
        wandb.log({
            # "elapsed_time": start_time - time.time(),
            "epoch": epoch,
            "batch": batch,
            "batch_loss": loss.item(),
            # "current_loss": cur_loss,
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
                    epoch, batch, len(train_data_batches) // max_seq_len, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# evaluation


def evaluate(model, data_source, config):
    max_seq_len = extract_config(config, "max_seq_len")

    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(max_seq_len).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, max_seq_len):
            data, targets = get_batch(max_seq_len, data_source, i)

            # print(data)
            # print(targets)
            if data.size(0) != max_seq_len:
                src_mask = model.generate_square_subsequent_mask(
                    data.size(0)).to(device)
            # output = model(data, targets)
            reshape_seq_len = min(data.size(0), max_seq_len)
            targets_flat = targets.reshape(
                reshape_seq_len, targets.size(0)//reshape_seq_len)
            output = model(data, src_mask)
            # output = model(data, targets_flat, src_mask, src_mask)
            # output = model(data, targets_flat, src_mask, src_mask)

            output_flat = output.view(-1, ntokens)
            loss = criterion(output_flat, targets)
            # update_artifact_loss(artifacts, 'training', 'CrossEntropyLoss', epoch, batch, loss.item())
            total_loss += len(data) * loss.item()

            wandb.log({
                # "elapsed_time": start_time - time.time(),
                # "epoch": epoch,
                "batch": i,
                "batch_loss": loss.item(),
                # "current_loss": cur_loss,
                "ppl": math.exp(loss.item()),
            })
    return total_loss / (len(data_source) - 1)

# sweep function


def train_and_eval():
    run = wandb.init(config=default_config)
    config = run.config
    print(config)

    # setup data
    # extract config vars
    embedding_dimension, n_attention_heads, n_encoder_layers, n_decoder_layers, ff_dimension, dropout, batch_size, eval_batch_size, learning_rate = extract_config(
        config, "embedding_dimension", "n_attention_heads", "n_encoder_layers", "n_decoder_layers", "ff_dimension", "dropout", "batch_size", "eval_batch_size", "learning_rate")

    # configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # load training data
    train_data, val_data, test_data, vocab = load_data(config)
    ntokens = len(vocab.stoi)

    # batch data
    train_data_batches = batchify(train_data, batch_size, device)
    val_data_batches = batchify(val_data, eval_batch_size, device)
    test_data_batches = batchify(test_data, eval_batch_size, device)

    # instantiate model
    model = DecoderOnlyTransformer(ntokens, d_model=embedding_dimension, nhead=n_attention_heads, num_encoder_layers=n_encoder_layers,
                                   num_decoder_layers=n_decoder_layers, dim_feedforward=ff_dimension, dropout=dropout).to(device)

    # hyperparams
    # lr = 5.0 # learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # train loop
    best_val_loss = float("inf")
    epochs = 3  # The number of epochs
    best_model = None
    artifacts = initalize_artifacts(
        config, train_data_batches, val_data_batches)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train(model, config, epoch, artifacts)
        val_loss = evaluate(model, val_data_batches, config)
        wandb.log({"val_loss": val_loss, "val_ppl": math.exp(
            val_loss), "epoch": epoch})
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    visualize_artifacts(artifacts)

    # test model
    test_loss = evaluate(best_model, test_data_batches, config)
    wandb.log({"test_loss": test_loss, "test_ppl": math.exp(test_loss)})

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    return best_model


default_config = {
    "embedding_dimension": 200,
    "ff_dimension": 200,
    "n_attention_heads": 2,
    "n_encoder_layers": 0,
    "n_decoder_layers": 2,
    "dataset": Dataset.PennTreebank.name,
    "segmentation": Segmentation.Character.name,
    "max_seq_len": 35,
    "batch_size": 20,
    "eval_batch_size": 10,
    "dropout": 0.2,
    "n_epochs": 3,
    "learning_rate": 0.5,
    "loss_criterion": "CrossEntropyLoss"
}

# train_and_eval()

if __name__ == "__main__":
  print("Run Sweep")
else:
  print("Functions to Run Sweep")
