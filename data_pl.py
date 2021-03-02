import time

import torch
from torch.utils.data import DataLoader

from torchtext import datasets
from torchtext.data import Field
from torchtext.data.utils import get_tokenizer

from transformers import AutoTokenizer

from utils import extract_config
from constants import *

import lineflow.datasets as lfds


class TextDataloader:
    def __init__(self, dataset, max_seq_len):
        self.max_seq_len = max_seq_len
        self.dataset = dataset
        self.dataset_len = len(dataset)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        i = self.index
        seq_len = min(self.max_seq_len, self.dataset_len - 1 - i)
        data = self.dataset[i:i+seq_len]
        target = self.dataset[i+1:i+1+seq_len].reshape(-1)
        self.index += 1
        return data, target

# load training data
def load_data(config):
    segmentation = extract_config(config, "segmentation")

    if segmentation == Segmentation.Word.name:
        return load_data_word(config)
    # if segmentation == Segmentation.Subword.name:
    #     return load_data_subword(config)
    # if segmentation == Segmentation.Character.name:
    #     return load_data_character(config)
    else:
        raise ValueError(f'Segementation {segmentation} not supported.')

# load word based training data
def load_data_word(config):
    print("[Start Load Data]")
    ts = time.time()

    # get dataset
    dataset, max_seq_len = extract_config(config, "dataset", "max_seq_len")
    dataset = getattr(datasets, dataset)
    print(f"Fetched Data ({time.time() - ts:3f}s)")

    # # tokenize
    tokenizer = get_tokenizer('basic_english')
    field_processor = Field(tokenize=tokenizer)

    # split dataset
    train_dataset, val_dataset, test_dataset = dataset.splits(
        text_field=field_processor)
    print(f"Tokenized and Split Data ({time.time() - ts:3f}s)")

    # get vocabulary
    field_processor.build_vocab(
        train_dataset, val_dataset, test_dataset, min_freq=1)
    vocab = field_processor.vocab
    print(f"Built Vocab ({time.time() - ts:3f}s)")

    # data prep
    def data_prep(tt_dataset_split):
        raw_text_iter = tt_dataset_split[0].text
        data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                                dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # setup dataloaders
    train_dataloader = TextDataloader(data_prep(train_dataset), max_seq_len)
    val_dataloader = TextDataloader(data_prep(val_dataset), max_seq_len)
    test_dataloader = TextDataloader(data_prep(test_dataset), max_seq_len)

    print(f"[End Load Data] ({time.time() - ts:3f}s)")
    return train_dataloader, val_dataloader, test_dataloader, vocab


if __name__ == "__main__":
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
        "learning_rate": 0.0001,
        "adam_b1": 0.9,
        "adam_b2": 0.999,
        "adam_l2_weightdecay": 0.01,
        "loss_criterion": "CrossEntropyLoss"
    }

    load_data_word(config)
