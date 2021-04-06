import time
import os

import torch
import math
import re
import string

import numpy as np

# from torch.utils.data import DataLoader

from torchtext import datasets
from torchtext.data import Field
from torchtext.data.utils import get_tokenizer

# from transformers import AutoTokenizer

# from transformers import GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

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

        # get seqence order
        num_seqs = (len(dataset) - 1) // self.max_seq_len
        self.seq_order = np.array(range(num_seqs))
        if shuffle:
            np.random.shuffle(self.seq_order)

        # get source, target datasets, trim
        self.dataset = dataset
        self.source = self.shuffle_dataset(dataset[0: len(dataset) - 1])
        self.targets = self.shuffle_dataset(dataset[1: len(dataset)])

        # fill remaining batch with beginning
        epoch_fill_count = (batch_size - num_seqs %
                            batch_size) * self.max_seq_len
        self.source = torch.cat([self.source, self.source[0:epoch_fill_count]])
        self.targets = torch.cat([self.targets, self.targets[0:epoch_fill_count]])

        self.dataset_len = len(self.source)
        self.num_batches = ((self.dataset_len - 1) //
                            self.max_seq_len) // self.batch_size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        i = self.index
        chunk_pos = i * self.chunk_len
        data = self.source[chunk_pos: chunk_pos + self.chunk_len]
        target = self.targets[chunk_pos: chunk_pos + self.chunk_len]

        num_batches = min(
            self.batch_size, (self.dataset_len - chunk_pos) // self.max_seq_len)

        if num_batches == 0:
            raise StopIteration

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

    def shuffle_dataset(self, dataset):
        shuffled_dataset = map(
            lambda x: dataset[x * self.max_seq_len: (x + 1) * self.max_seq_len], self.seq_order)
        return torch.cat(list(shuffled_dataset))


def split_dataset(config):
    dataset  = extract_config(config, "dataset")
    location = TRAINING_DATA[dataset]['location']
    paths = list(map(lambda x: str(DATA_PATH+location+x),
                     TRAINING_DATA[dataset]['filenames']))

    # train data
    train_data = []
    valid_data = []
    test_data = []
    for path in paths:
        raw_data = list(open(path, newline='\n'))
        raw_data = list(filter(lambda x: x != '\n', raw_data))
        if re.search("train", path):
           train_data = raw_data
        if re.search("valid", path):
           valid_data = raw_data
        if re.search("test", path):
           test_data = raw_data

    return train_data, valid_data, test_data

# load training data
def load_data(config):
    segmentation = extract_config(config, "segmentation")

    if segmentation == Segmentation.Word.name:
        return load_data_general(config)
    if segmentation == Segmentation.Subword.name:
        return load_data_general(config)
    if segmentation == Segmentation.BBPE.name:
        return load_data_general(config)
    if segmentation == Segmentation.Character.name:
        return load_data_general(config)
    else:
        raise ValueError(f'Segementation {segmentation} not supported.')

# load subword based training data
def create_subword_tokenizer(config):
    dataset, vocab_size, segmentation = extract_config(
        config, "dataset", "vocab_size", "segmentation")

    # get location
    output_location = 'tokenizer/'
    tokenizer_loc = segmentation +'_tokenizer_' + str(dataset) + '_' + str(vocab_size) + ".tokenizer.json"
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
    dataset, vocab_size, segmentation = extract_config(config, "dataset", "vocab_size", "segmentation")

    # get location
    output_location = 'tokenizer/'
    tokenizer_loc = segmentation +'_tokenizer_' + str(dataset) + ".tokenizer.json"
    path_to_tokenizer_loc = DATA_PATH+output_location
    tokenizer_filepath = path_to_tokenizer_loc+tokenizer_loc

    # load tokenizer
    if os.path.isfile(tokenizer_filepath):
        tokenizer = Tokenizer.from_file(tokenizer_filepath)
        return tokenizer

    location = TRAINING_DATA[dataset]['location']
    paths = list(map(lambda x: str(DATA_PATH+location+x),
                     TRAINING_DATA[dataset]['filenames']))
    # train tokenizer
    tokenizer.train(files=paths,
                    vocab_size=vocab_size,
                    min_frequency=2,
                    special_tokens=["<|endoftext|>"])
    tokenizer.enable_truncation(max_length=1024)
 
    # save tokenizer
    try:
        if not os.path.isdir(path_to_tokenizer_loc):
            os.makedirs(path_to_tokenizer_loc)
        tokenizer.save(str(tokenizer_filepath))
    except Exception as e:
        print("Error saving tokenizer", e)

    return tokenizer

def create_word_tokenizer(config):
    dataset, vocab_size, segmentation = extract_config(
        config, "dataset", "vocab_size", "segmentation")

    # get location
    output_location = 'tokenizer/'
    tokenizer_loc = segmentation +'_tokenizer_' + str(dataset) + ".tokenizer.json"
    path_to_tokenizer_loc = DATA_PATH+output_location
    tokenizer_filepath = path_to_tokenizer_loc+tokenizer_loc

    # load tokenizer
    if os.path.isfile(tokenizer_filepath):
        tokenizer = Tokenizer.from_file(tokenizer_filepath)
        return tokenizer

    # build tokenizer
    tokenizer = Tokenizer(WordLevel())
    tokenizer.pre_tokenizer = Whitespace()

    location = TRAINING_DATA[dataset]['location']
    paths = list(map(lambda x: str(DATA_PATH+location+x),
                     TRAINING_DATA[dataset]['filenames']))
  
    trainer = WordLevelTrainer(
        min_frequency=1,
        # vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<unk>", "<eos>"]
        )

    tokenizer.train(files=paths, trainer=trainer)

    # save tokenizer
    try:
        if not os.path.isdir(path_to_tokenizer_loc):
            os.makedirs(path_to_tokenizer_loc)
        tokenizer.save(str(tokenizer_filepath))
    except Exception as e:
        print("Error saving tokenizer", e)

    return tokenizer

class CharacterTokenizer:
    def __init__(self):
        self.vocab = {}
        for x in str.encode(string.printable):
            self.vocab[x + 2] = chr(x)

    def encode(self, str, is_pretokenized=True):
        if is_pretokenized:
            str = " ".join(str)
        tokens = [chr(x) for x in str.encode(str)]
        ids = [x + 2 for x in str.encode(str)]
        
        # sad hack
        class Object(object):
            pass
        data = Object()

        data.tokens = tokens
        data.ids = ids

        return data

    def decode(self, tokens):
        return "".join([chr(x - 2) if x > 1 else "" for x in tokens])

    def get_vocab(self):
        return self.vocab

# load data using huggingface tokenization
# (word, subword, bbpe)
def load_data_general(config):
    print("[Start Load Data]")
    ts = time.time()

    # get dataset
    dataset, batch_size, max_seq_len, segmentation, torchtext_split = extract_config(
        config, "dataset", "batch_size", "max_seq_len", "segmentation", "torchtext_split")
    tt_dataset = getattr(datasets, dataset)
    print(f"Fetched Data ({time.time() - ts:3f}s)")

    # split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(config)
    if torchtext_split:
        train_dataset, val_dataset, test_dataset = tt_dataset.splits(
            text_field=Field())
    print(f"Tokenized and Split Data ({time.time() - ts:3f}s)")

    # tokenize
    if segmentation == Segmentation.Subword.name:
        tokenizer = create_subword_tokenizer(config)
    elif segmentation == Segmentation.BBPE.name:
        tokenizer = create_bbpe_tokenizer(config)
    elif segmentation == Segmentation.Word.name:
        tokenizer = create_word_tokenizer(config)
    elif segmentation == Segmentation.Character.name:
        tokenizer = CharacterTokenizer()

    # get vocabulary
    vocab = tokenizer.get_vocab()

    # prep data
    def prep_data(dataset_arr):
        if torchtext_split:
            raw_text_iter = dataset_arr[0].text
            data = torch.tensor(tokenizer.encode(raw_text_iter, is_pretokenized=True).ids, dtype=torch.long)
            return data

        raw_text_iter = dataset_arr
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
        "dataset": Dataset.PennTreebank.name,
        "segmentation": Segmentation.Subword.name,
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
        "loss_criterion": "CrossEntropyLoss",
        "torchtext_split": True,
    }

    train_dataloader, val_dataloader, test_dataloader, vocab, tokenizer = load_data(config)
    # print(vocab.stoi)
    
    for batch in train_dataloader:
        # print(batch)    
        data, targets = batch
        print("[train] data.shape - targets.shape: ", data.shape,  targets.shape)
        print(tokenizer.decode(data[0].tolist()))
        print(tokenizer.decode(targets[0:len(data[0])].tolist()))
        break

    # for batch in val_dataloader:
    #     # print(batch)    
    #     data, targets = batch
    #     print("[val] data.shape - targets.shape: ", data.shape,  targets.shape)

    # print(data)
    # print(tokenizer.decode(data[0].tolist()))
    # print(tokenizer.decode(targets[0:20].tolist()))

    # check word 
