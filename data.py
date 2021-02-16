import time

import torch

from torchtext import datasets
from torchtext.data import Field
from torchtext.data.utils import get_tokenizer

from transformers import AutoTokenizer

from utils import extract_config
from constants import *

# data loading
def char_tokenizer(string):
    return [x for x in string]


def char_decoder(tokens):
    return "".join([x for x in tokens])

# batch functions
def batchify(data, bsz, device):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(max_seq_len, source, i):
    seq_len = min(max_seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# load training data
def load_data(config):
    segmentation = extract_config(config, "segmentation")

    if segmentation == Segmentation.Word.name:
        return load_data_word(config)
    if segmentation == Segmentation.Subword.name:
        return load_data_subword(config)
    if segmentation == Segmentation.Character.name:
        return load_data_character(config)
    else:
        raise ValueError(f'Segementation {segmentation} not supported.')

# load word based training data
def load_data_word(config):
    print("[Start Load Data]")
    ts = time.time()

    # get dataset
    dataset, segmentation = extract_config(config, "dataset", "segmentation")
    dataset = getattr(datasets, dataset)
    print(f"Fetched Data ({time.time() - ts:3f}s)")

    # tokenize
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

    def data_process(tt_dataset_split):
        raw_text_iter = tt_dataset_split[0].text
        data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                             dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(train_dataset)
    val_data = data_process(val_dataset)
    test_data = data_process(test_dataset)

    print(f"[End Load Data] ({time.time() - ts:3f}s)")
    return train_data, val_data, test_data, vocab


# load subword based training data
def load_data_subword(config):
    print("[Start Load Data]")
    ts = time.time()

    # get dataset
    dataset, segmentation = extract_config(config, "dataset", "segmentation")
    dataset = getattr(datasets, dataset)

    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
    field_processor = Field(tokenize=tokenizer.encode)

    # tokenizer = get_tokenizer('subword')
    # field_processor = Field(tokenize=tokenizer)
    print(f"Fetched Data ({time.time() - ts:3f}s)")

    # split dataset
    train_dataset, val_dataset, test_dataset = dataset.splits(
        text_field=field_processor)
    print(f"Split Data ({time.time() - ts:3f}s)")

    print(train_dataset)
    # get vocabulary
    # field_processor.build_vocab(train_dataset, val_dataset, test_dataset, min_freq=1)
    # vocab = field_processor.vocab
    vocab = tokenizer.get_vocab()

    print(f"Build Vocab ({time.time() - ts:3f}s)")

    def data_process(tt_dataset_split):
        raw_text_iter = tt_dataset_split[0].text
        data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                             dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(train_dataset)
    val_data = data_process(val_dataset)
    test_data = data_process(test_dataset)

    print(f"[End Load Data] ({time.time() - ts:3f}s)")
    return train_data, val_data, test_data, vocab


# load character based training data
def load_data_character(config):
    print("[Start Load Data]")
    ts = time.time()

    # get dataset
    dataset, segmentation = extract_config(config, "dataset", "segmentation")
    dataset = getattr(datasets, dataset)
    # tokenizer = get_tokenizer('basic_english')
    tokenizer = char_tokenizer
    field_processor = Field(tokenize=tokenizer)
    print(f"Fetched Data ({time.time() - ts:3f}s)")

    # split dataset
    train_dataset, val_dataset, test_dataset = dataset.splits(
        text_field=field_processor)
    print(f"Split Data ({time.time() - ts:3f}s)")

    print(train_dataset[0:10])
    # get vocabulary
    field_processor.build_vocab(
        train_dataset, val_dataset, test_dataset, min_freq=1)
    vocab = field_processor.vocab
    print(f"Build Vocab ({time.time() - ts:3f}s)")

    def data_process(tt_dataset_split):
        raw_text_iter = tt_dataset_split[0].text
        data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                             dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(train_dataset)
    val_data = data_process(val_dataset)
    test_data = data_process(test_dataset)

    print(f"[End Load Data] ({time.time() - ts:3f}s)")
    return train_data, val_data, test_data, vocab
