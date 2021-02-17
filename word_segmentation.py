from words2bytes import train_and_eval
from constants import *

config = {
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

# benchmarked against https://arxiv.org/pdf/1904.09408v2.pdf
# bert_lm_12_768_12_300_1150_wikitext2
benchmark_config_1 = {
    "embedding_dimension": 768,  # units
    "ff_dimension": 3072,  # hidden_size
    "n_attention_heads": 12,  # num_heads
    "n_decoder_layers": 12,  # num_layers
    "dataset": Dataset.WikiText2.name,
    "segmentation": Segmentation.Word.name,
    "max_seq_len": 512,  # max_length
    "dropout": 0.1,  # dropout
    "batch_size": 16,
    "eval_batch_size": 8,
    "n_epochs": 6,
    "learning_rate": 5.0,
    "loss_criterion": "CrossEntropyLoss"
}

train_and_eval(benchmark_config_1, "skgbafa")
