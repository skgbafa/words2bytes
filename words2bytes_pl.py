# imports
import wandb

import pytorch_lightning as pl

from constants import *

from data import load_data
from transformer import DecoderOnlyTransformer


# default config
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
    "learning_rate": 0.0001,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "adam_l2_weightdecay": 0.01,
    "loss_criterion": "CrossEntropyLoss"
}

# experiment generation
def generateExperiements():
    WANDB_ENTITY = "skgbafa"
    # WANDB_ENTITY = "openai-scholars"

    # experiment_datasets = [ Dataset.PennTreebank.name, Dataset.WikiText2.name, Dataset.WikiText103.name ]
    experiment_datasets = [ Dataset.WikiText2.name ]
    experiment_segmentation = [ Segmentation.Word.name, Segmentation.Character.name ]

    sweep_parameters = {
        "n_attention_heads": {
            "values": [2, 3]
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

# sweep function
def train_and_eval(config=default_config, entity=WANDB_ENTITY, num_gpus=2):
    run = wandb.init(config=config, entity=entity)
    config = run.config

    # load data
    train_loader, val_loader, test_loader, vocab = load_data(config)
    ntokens = len(vocab.stoi)

    # run model
    model = DecoderOnlyTransformer(config, ntokens)
    trainer = pl.Trainer(gpus=num_gpus, accelerator="dp")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
  print("Run Sweep")
  # set up experiments
  sweep_id = generateExperiements()

  # run experiments
  wandb.agent(sweep_id, function=train_and_eval)

