from words2bytes_pl import train_and_eval
from constants import *

import wandb
import numpy as np
import pytorch_lightning as pl

from constants import *
from utils import *

from data_pl import load_data
from transformer_pl import DecoderOnlyTransformer

# benchmarked against https://arxiv.org/pdf/1904.09408v2.pdf
# bert_lm_12_768_12_300_1150_wikitext2
benchmark_config_1 = {
    "embedding_dimension": 768,  # units
    # "ff_dimension": 3072,  # hidden_size
    "ff_dimension": 768,  # hidden_size
    "n_attention_heads": 12,  # num_heads
    "n_encoder_layers": 0,  # num_layers
    "n_decoder_layers": 12,  # num_layers
    "dataset": Dataset.PennTreebank.name,
    "segmentation": Segmentation.Subword.name,
    "vocab_size": 40000,
    "max_seq_len": 64,  # max_length
    "dropout": 0.1,  # dropout
    # "batch_size": 16,
    "batch_size": 12,
    "eval_batch_size": 8,
    "n_epochs": 10,
    # "learning_rate": 0.0000625,
    "learning_rate": 0.00003125,
    # "learning_rate": 0.000015625,
    # "learning_rate": 0.00000625,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "adam_l2_weightdecay": 0.01,
    "loss_criterion": "CrossEntropyLoss"
}


def train_and_eval(config=benchmark_config_1, entity=WANDB_ENTITY, num_gpus=4):
    run = wandb.init(config=config, entity=entity)
    config = run.config
    n_epochs = extract_config(config, "n_epochs")

    # load data
    train_loader, val_loader, test_loader, vocab, tokenizer = load_data(config)
    ntokens = len(vocab)

    # run model
    model = DecoderOnlyTransformer(config, ntokens)
    trainer = pl.Trainer(gpus=num_gpus, accelerator="dp", max_epochs=n_epochs)
    trainer.fit(model, train_loader, val_loader)


# train_and_eval()

original_lr = 0.0000625
# scale = [0.7, 0.5, 0.33, 0.25, 0.1]
scale = [0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.375, 0.35]

sweep_parameters = {
    "learning_rate": {
        "values": np.multiply(original_lr, scale).tolist()
    }
}

sweep_config = {
    "name": "LR Sweeps",
    "method": "grid",
    "parameters": sweep_parameters
}

sweep_id = wandb.sweep(sweep_config, entity=WANDB_ENTITY)
wandb.agent(sweep_id, function=train_and_eval)
