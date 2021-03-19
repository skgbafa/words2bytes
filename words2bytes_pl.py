# imports
import wandb

import pytorch_lightning as pl

from constants import *
from utils import *

from data_pl import load_data
from transformer_pl import DecoderOnlyTransformer


# default config
default_config = {
    "embedding_dimension": 200,
    "ff_dimension": 200,
    "n_attention_heads": 2,
    "n_encoder_layers": 0,
    "n_decoder_layers": 2,
    "dataset": Dataset.PennTreebank.name,
    "segmentation": Segmentation.Word.name,
    "vocab_size": 40000, # subword/bbpe only
    "max_seq_len": 35,
    "batch_size": 20,
    "eval_batch_size": 10,
    "dropout": 0.2,
    "n_epochs": 3,
    "learning_rate": 0.0001,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "adam_l2_weightdecay": 0.01,
    "loss_criterion": "CrossEntropyLoss"
}

# benchmarked against https://arxiv.org/pdf/1904.09408v2.pdf
# bert_lm_12_768_12_300_1150_wikitext2
benchmark_config_1 = {
    "embedding_dimension": 768,  # units
    "ff_dimension": 3072,  # hidden_size
    "n_attention_heads": 12,  # num_heads
    "n_encoder_layers": 0,  # num_layers
    "n_decoder_layers": 12,  # num_layers
    "dataset": Dataset.PennTreebank.name,
    "segmentation": Segmentation.Word.name,
    "vocab_size": 40000,
    "max_seq_len": 64,  # max_length
    "dropout": 0.1,  # dropout
    "batch_size": 8,
    "eval_batch_size": 8,
    "n_epochs": 3,
    "learning_rate": 0.00003125,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "adam_l2_weightdecay": 0.01,
    "loss_criterion": "CrossEntropyLoss"
}

# experiment generation
def generateExperiements():
    # WANDB_ENTITY = "skgbafa"
    WANDB_ENTITY = "openai-scholars"

    experiment_datasets = [ Dataset.PennTreebank.name,Dataset.WikiText2.name, Dataset.WikiText103.name ]
    # experiment_datasets = [ Dataset.WikiText2.name ]
    experiment_segmentation = [ Segmentation.Word.name, Segmentation.Subword.name ]

    sweep_parameters = {
        "dataset": {
            "values": experiment_datasets
        },
        "n_epochs": {
            "values": [20]
        },
        "segmentation": {
            "values": experiment_segmentation
        }
    }

    sweep_config = {
        "name": "Benchmark Sweeps",
        "method": "grid",
        "parameters": sweep_parameters
    }

    sweep_id = wandb.sweep(sweep_config, entity=WANDB_ENTITY)

    return sweep_id

# sweep function
def train_and_eval(config=benchmark_config_1, entity=WANDB_ENTITY, num_gpus=4):
    run = wandb.init(config=config, entity=entity)
    config = run.config
    n_epochs = extract_config(config, "n_epochs")

    # load data
    train_loader, val_loader, test_loader, vocab, tokenizer = load_data(config)
    ntokens = len(vocab)

    # run model
    trainer = pl.Trainer(gpus=num_gpus, accelerator="dp", max_epochs=n_epochs)
    model = DecoderOnlyTransformer(config, ntokens, trainer)
    trainer.fit(model, train_loader, val_loader)
        # trainer.test(model, test_loader)



if __name__ == "__main__":
  print("Run Sweep")
  # set up experiments
  sweep_id = generateExperiements()

  # run experiments
  wandb.agent(sweep_id, function=train_and_eval)

