import time
import math
import torch
import torch.nn as nn

import wandb

from artifacts import initalize_artifacts, visualize_artifacts
from transformer import DecoderOnlyTransformer
from constants import *
from data import load_data, batchify
from utils import extract_config
from training import train, evaluate

# word based config
default_config = {
    "embedding_dimension": 200,
    "ff_dimension": 200,
    "n_attention_heads": 2,
    "n_encoder_layers": 0,
    "n_decoder_layers": 2,
    "dataset": Dataset.PennTreebank.name,
    "segmentation": Segmentation.Word.name,
    "max_seq_len": 35,
    "batch_size": 20,
    "eval_batch_size": 10,
    "dropout": 0.2,
    "n_epochs": 3,
    "learning_rate": 5.0,
    "loss_criterion": "CrossEntropyLoss"
}


# testing word based model
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# runtime vars
runtime = {
    "criterion": criterion,
    "optimizer": optimizer,
    "scheduler": scheduler,
    "ntokens": ntokens,
    "device": device,
}

# train loop
best_val_loss = float("inf")
epochs = 3  # The number of epochs
best_model = None
artifacts = initalize_artifacts(
    config, train_data_batches, val_data_batches)

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    train(model, train_data_batches, config, runtime, epoch, artifacts)
    val_loss = evaluate(model, val_data_batches, config, runtime)
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
test_loss = evaluate(best_model, test_data_batches, config, runtime)
wandb.log({"test_loss": test_loss, "test_ppl": math.exp(test_loss)})

print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
