import time
import os

import torch
import math

import numpy as np

# from torch.utils.data import DataLoader

from torchtext import datasets
from torchtext.data import Field
from torchtext.data.utils import get_tokenizer

# from transformers import AutoTokenizer

# from transformers import GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from utils import extract_config
from constants import *
from utils import *


class TextDataloader:
    def __init__(self, dataset, max_seq_len, batch_size, shuffle=True):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        # shuffle logic vars
        self.shuffle = shuffle
        self.chunk_len = max_seq_len * batch_size

        # trim dataset, fix for multigpu batching bugs
        num_batches = math.ceil(len(dataset)/self.chunk_len)
        trimmed_dataset_size = (num_batches - 1) * self.chunk_len + 1
        self.dataset = dataset[0: trimmed_dataset_size]
        self.dataset_len = trimmed_dataset_size

        # non-shuffled batch order
        self.batch_order = np.array(range(num_batches))

        if shuffle:
            np.random.shuffle(self.batch_order)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index > len(self.batch_order) - 1:
            raise StopIteration

        i = self.batch_order[self.index]
        chunk_pos = i * self.chunk_len
        data = self.dataset[chunk_pos: chunk_pos + self.chunk_len]
        target = self.dataset[(chunk_pos) + 1: (chunk_pos + self.chunk_len) + 1]

        num_batches = min(self.batch_size, (self.dataset_len - chunk_pos) // self.max_seq_len)
        if num_batches == 0:
            raise StopIteration

        if(len(data) != len(target)):
            # remove mismatched batch sizes
            data = data.narrow(0, 0, self.max_seq_len * (num_batches - 1))
            target = target.narrow(0, 0, self.max_seq_len * (num_batches - 1))

        self.index += 1

        return self.batchify(data, target, num_batches)

    def batchify(self, data, target, num_batches):
        # Evenly divide the data across the batch_size batches.
        data = data.view(num_batches, -1).contiguous()
        target = target.view(num_batches, -1).contiguous()

        # shuffle data
        if self.shuffle:
            permutation = torch.randperm(data.size(0))
            data = data[permutation]
            target = target[permutation]

        # flatten targets
        target = target.reshape(-1)
        return data, target.reshape(-1)

# load training data
def load_data(config):
    segmentation = extract_config(config, "segmentation")

    if segmentation == Segmentation.Word.name:
        return load_data_word(config)
    if segmentation == Segmentation.Subword.name:
        return load_data_subword(config)
    if segmentation == Segmentation.BBPE.name:
        return load_data_subword(config)
    # if segmentation == Segmentation.Character.name:
    #     return load_data_character(config)
    else:
        raise ValueError(f'Segementation {segmentation} not supported.')

# load word based training data
def load_data_word(config):
    print("[Start Load Data]")
    ts = time.time()

    # get dataset
    dataset, batch_size, max_seq_len = extract_config(config, "dataset", "batch_size", "max_seq_len")
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
        data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
        return data

    # setup dataloaders
    train_dataloader = TextDataloader(data_prep(train_dataset), max_seq_len, batch_size)
    val_dataloader = TextDataloader(data_prep(val_dataset), max_seq_len, batch_size)
    test_dataloader = TextDataloader(data_prep(test_dataset), max_seq_len, batch_size)

    print(f"[End Load Data] ({time.time() - ts:3f}s)")
    return train_dataloader, val_dataloader, test_dataloader, vocab, tokenizer

# load subword based training data
def create_subword_tokenizer(config):
    dataset, vocab_size = extract_config(
        config, "dataset", "vocab_size")

    # get location
    output_location = 'tokenizer/'
    tokenizer_loc = 'bpe_tokenizer_' + str(dataset) + '_' + str(vocab_size) + ".tokenizer.json"
    path_to_tokenizer_loc = DATA_PATH+output_location
    tokenizer_filepath = path_to_tokenizer_loc+tokenizer_loc

    # load tokenizer
    if os.path.isfile(tokenizer_filepath):
        tokenizer = Tokenizer.from_file(tokenizer_filepath)
        return tokenizer

    # build tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    location = TRAINING_DATA[dataset]['location']
    paths = list(map(lambda x: str(DATA_PATH+location+x),
                     TRAINING_DATA[dataset]['filenames']))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<unk>", "<eos>"])

    tokenizer.train(files=paths, trainer=trainer)

    # save tokenizer
    try:
        if not os.path.isdir(path_to_tokenizer_loc):
            os.makedirs(path_to_tokenizer_loc)
        tokenizer.save(str(tokenizer_filepath))
    except Exception as e:
        print("Error saving tokenizer", e)

    return tokenizer

def create_bbpe_tokenizer(config):
    tokenizer = ByteLevelBPETokenizer()

    # prep data
    dataset, vocab_size = extract_config(config, "dataset", "vocab_size")
    data_path = './.data/'
    location = TRAINING_DATA[dataset]['location']
    paths = list(map(lambda x: str(data_path+location+x),
                     TRAINING_DATA[dataset]['filenames']))
    # train tokenizer
    tokenizer.train(files=paths,
                    vocab_size=vocab_size,
                    min_frequency=2,
                    special_tokens=["<|endoftext|>"])
    tokenizer.enable_truncation(max_length=1024)

    return tokenizer

def load_data_subword(config):
    print("[Start Load Data]")
    ts = time.time()

    # get dataset
    dataset, batch_size, max_seq_len, segmentation = extract_config(
        config, "dataset", "batch_size", "max_seq_len", "segmentation")
    dataset = getattr(datasets, dataset)
    print(f"Fetched Data ({time.time() - ts:3f}s)")

    # split dataset
    train_dataset, val_dataset, test_dataset = dataset.splits(
        text_field=Field())
    print(f"Tokenized and Split Data ({time.time() - ts:3f}s)")

    # tokenize
    if segmentation == Segmentation.Subword.name:
        tokenizer = create_subword_tokenizer(config)
    elif segmentation == Segmentation.BBPE.name:
        tokenizer = create_bbpe_tokenizer(config)

    # get vocabulary
    vocab = tokenizer.get_vocab()

    # prep data
    def prep_data(dataset):
        raw_text_iter = dataset[0].text
        data = [torch.tensor(tokenizer.encode(item).ids,
                             dtype=torch.long) for item in raw_text_iter]
        data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
        return data

    # setup dataloaders
    train_dataloader = TextDataloader(prep_data(train_dataset), max_seq_len, batch_size)
    val_dataloader = TextDataloader(prep_data(val_dataset), max_seq_len, batch_size)
    test_dataloader = TextDataloader(prep_data(test_dataset), max_seq_len, batch_size)

    print(f"[End Load Data] ({time.time() - ts:3f}s)")
    return train_dataloader, val_dataloader, test_dataloader, vocab, tokenizer

if __name__ == "__main__":
    config = {
        "embedding_dimension": 200,
        "ff_dimension": 200,
        "n_attention_heads": 2,
        "n_encoder_layers": 0,
        "n_decoder_layers": 2,
        "dataset": Dataset.WikiText2.name,
        "segmentation": Segmentation.Word.name,
        "vocab_size": 40000,
        "max_seq_len": 32,
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

    train_dataloader, val_dataloader, test_dataloader, vocab, tokenizer = load_data(config)
    # print(vocab.stoi)
    
    for batch in train_dataloader:
        # print(batch)    
        data, targets = batch
        print("[train] data.shape - targets.shape: ", data.shape,  targets.shape)

    # for batch in val_dataloader:
    #     # print(batch)    
    #     data, targets = batch
    #     print("[val] data.shape - targets.shape: ", data.shape,  targets.shape)

    # print(data)
    # print(tokenizer.decode(data[0].tolist()))
    # print(tokenizer.decode(targets[0:20].tolist()))

    # check word 
