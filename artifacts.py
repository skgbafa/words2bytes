import torch
import math

import numpy as np 
import matplotlib.pyplot as plt

from utils import extract_config

# generate/visualize artifacts
def initalize_artifacts(config, train_data_batches, val_data_batches):
    n_epochs, max_seq_len = extract_config(config, "n_epochs", "max_seq_len")
    training_cel = torch.ones(n_epochs, math.ceil(
        len(train_data_batches) / max_seq_len)) * float("inf")
    validation_cel = torch.ones(n_epochs, math.ceil(
        len(val_data_batches) / max_seq_len)) * float("inf")
    artifacts = {
        "training": {
            "CrossEntropyLoss": training_cel
        },
        "validation": {
            "CrossEntropyLoss": validation_cel
        }
    }
    return artifacts


def update_artifact_loss(artifacts, training_stage, metric, epoch, batch, value):
    try:
        artifacts[training_stage][metric][epoch - 1][batch] = value
    except Exception as e:
        print("exception:", e)
        print("epoch", epoch)
        print("batch", batch)
        print(artifacts)


def visualize_artifacts(artifacts):
    flat_loss = artifacts['training']['CrossEntropyLoss'].reshape(-1)
    count = flat_loss.size(0)
    batch_number = np.arange(0, flat_loss.size(0))
    plt.plot(batch_number, flat_loss)
    plt.legend("CrossEntropyLoss")
    None


if __name__ == "__main__":
  from data import load_data, batchify

  # setup
  config = {
    "embedding_dimension": 200,
    "ff_dimension": 200,
    "n_attention_heads": 2,
    "n_encoder_layers": 0,
    "n_decoder_layers": 2,
    "dataset": "PennTreebank",
    "segmentation": "Word",
    "max_seq_len": 35,
    "batch_size": 20,
    "eval_batch_size": 10,
    "dropout": 0.2,
    "n_epochs": 3,
    "learning_rate": 0.5,
    "loss_criterion": "CrossEntropyLoss"
  }

  # extract config
  batch_size, eval_batch_size = extract_config(config, "batch_size", "eval_batch_size")

  # configure device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load training data
  train_data, val_data, test_data, vocab = load_data(config)
  ntokens = len(vocab.stoi)

  # batch data
  train_data_batches = batchify(train_data, batch_size, device)
  val_data_batches = batchify(val_data, eval_batch_size, device)
  test_data_batches = batchify(test_data, eval_batch_size, device)


  # testing
  artifacts = initalize_artifacts(config, train_data_batches, val_data_batches)
  update_artifact_loss(artifacts, 'training', 'CrossEntropyLoss', 0, 1, 0.5)
  update_artifact_loss(artifacts, 'training', 'CrossEntropyLoss', 0, 2, 3)
  # artifacts['training']['CrossEntropyLoss'].reshape(-1)
  visualize_artifacts(artifacts)
  # visualize_artifacts(artifacts)
  print("artifacts.py run complete")
