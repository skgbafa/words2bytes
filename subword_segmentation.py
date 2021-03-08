from words2bytes_pl import train_and_eval
from constants import *

# benchmarked against https://arxiv.org/pdf/1904.09408v2.pdf
# bert_lm_12_768_12_300_1150_wikitext2
benchmark_config_1 = {
    "embedding_dimension": 768,  # units
    "ff_dimension": 3072,  # hidden_size
    "n_attention_heads": 12,  # num_heads
    "n_encoder_layers": 0,  # num_layers
    "n_decoder_layers": 12,  # num_layers
    "dataset": Dataset.PennTreebank.name,
    "segmentation": Segmentation.Subword.name,
    "vocab_size": 40000,
    "max_seq_len": 64,  # max_length
    "dropout": 0.1,  # dropout
    "batch_size": 16,
    "eval_batch_size": 8,
    "n_epochs": 10,
    "learning_rate": 0.0000625,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "adam_l2_weightdecay": 0.01,
    "loss_criterion": "CrossEntropyLoss"
}

train_and_eval(benchmark_config_1, num_gpus=4)
