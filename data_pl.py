import time
import os

import torch
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
    def __init__(self, dataset, tokenizer, max_seq_len, batch_size):
        self.dataset = self.prep_data(dataset, tokenizer)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dataset_len = len(self.dataset)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        i = self.index

        seq_len = min(self.max_seq_len, self.dataset_len - 1 - i)
        chunk_len = seq_len * self.batch_size
        if (i > self.dataset_len // chunk_len):
            # end iteration
            raise StopIteration
        data = self.dataset[i:i + chunk_len]
        target = self.dataset[i+1:i+1+chunk_len].reshape(-1)

        self.index += 1
        data = self.batchify(data)
        return data, target

    def prep_data(self, dataset, tokenizer):
        raw_text_iter = dataset[0].text
        data = [torch.tensor(tokenizer.encode(item).ids,
                             dtype=torch.long) for item in raw_text_iter]
        data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
        return data
    
    def batchify(self, data):
        # Divide the dataset into batch_size parts.
        nbatch = data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * self.batch_size)
        # Evenly divide the data across the batch_size batches.
        data = data.view(-1, self.batch_size).contiguous()
        return data

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
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches.
        data = data.view(batch_size, -1).t().contiguous()
        return data

    # setup dataloaders
    train_dataloader = TextDataloader(data_prep(train_dataset), max_seq_len)
    val_dataloader = TextDataloader(data_prep(val_dataset), max_seq_len)
    test_dataloader = TextDataloader(data_prep(test_dataset), max_seq_len)

    print(f"[End Load Data] ({time.time() - ts:3f}s)")
    return train_dataloader, val_dataloader, test_dataloader, vocab, tokenizer

# load subword based training data
def create_subword_tokenizer(config):
    dataset, vocab_size = extract_config(
        config, "dataset", "vocab_size")
    
    # get location
    output_location = 'tokenizer/'
    tokenizer_loc = 'BBPE_tokenizer_' + str(vocab_size)
    path_to_tokenizer_loc = DATA_PATH+output_location+tokenizer_loc + '/'

    # # load tokenizer
    # tokenizer_pt = AutoTokenizer.from_pretrained(
    #     str(path_to_tokenizer_loc),
    #     pad_token='<|endoftext|>')
    # print(tokenizer_pt)

    # build tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    location = TRAINING_DATA[dataset]['location']
    paths = list(map(lambda x: str(DATA_PATH+location+x),
                     TRAINING_DATA[dataset]['filenames']))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<unk>"])

    tokenizer.train(files=paths, trainer=trainer)

    # # save tokenizer
    # try:
    #     if not os.path.isdir(path_to_tokenizer_loc):
    #         os.makedirs(path_to_tokenizer_loc)
    #     tokenizer.save_model(str(path_to_tokenizer_loc))
    # except Exception as e:
    #         print("Error saving tokenizer", e)

    return tokenizer

def create_bbpe_tokenizer(config):
    tokenizer = ByteLevelBPETokenizer()

    # prep data
    dataset, vocab_size, max_seq_len = extract_config(
        config, "dataset", "vocab_size", "max_seq_len")
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

    # setup dataloaders
    train_dataloader = TextDataloader(train_dataset, tokenizer, max_seq_len, batch_size)
    val_dataloader = TextDataloader(val_dataset, tokenizer, max_seq_len, batch_size)
    test_dataloader = TextDataloader(test_dataset, tokenizer, max_seq_len, batch_size)

    print(f"[End Load Data] ({time.time() - ts:3f}s)")
    return train_dataloader, val_dataloader, test_dataloader, vocab, tokenizer

# load bbpe tokenizer
# def create_bbpe_tokenizer(config):
#     tokenizer = ByteLevelBPETokenizer()

#     # prep data
#     dataset, vocab_size = extract_config(config, "dataset", "vocab_size")
#     data_path = './.data/'
#     location = TRAINING_DATA[dataset]['location']
#     paths = list(map(lambda x: str(data_path+location+x),
#                      TRAINING_DATA[dataset]['filenames']))
#     # train tokenixer
#     tokenizer.train(files=paths,
#         vocab_size=vocab_size,
#         min_frequency=2,
#         special_tokens=["<|endoftext|>"])
#     tokenizer.enable_truncation(max_length=1024)

#     return tokenizer

# def load_data_bbpe(config):
#     print("[Start Load Data]")
#     ts = time.time()

#     # get dataset
#     dataset, batch_size, max_seq_len = extract_config(
#         config, "dataset", "batch_size", "max_seq_len")
#     dataset = getattr(datasets, dataset)
#     print(f"Fetched Data ({time.time() - ts:3f}s)")

#     # tokenize
#     pretrained_weights = 'gpt2'
#     tokenizer_en = GPT2TokenizerFast.from_pretrained(pretrained_weights)
#     tokenizer_en.pad_token = tokenizer_en.eos_token

#     # split dataset
#     train_dataset, val_dataset, test_dataset = dataset.splits(
#         text_field=field_processor)
#     print(f"Tokenized and Split Data ({time.time() - ts:3f}s)")

#     # get vocabulary
#     vocab = tokenizer_en.get_vocab()

#     # data prep
#     def data_prep(tt_dataset_split):
#         raw_text_iter = tt_dataset_split[0].text
#         data = [tokenizer_en.encode(item).ids for item in raw_text_iter]
#         data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
#         # Divide the dataset into bsz parts.
#         nbatch = data.size(0) // batch_size
#         # Trim off any extra elements that wouldn't cleanly fit (remainders).
#         data = data.narrow(0, 0, nbatch * batch_size)
#         # Evenly divide the data across the batch_size batches.
#         data = data.view(batch_size, -1).t().contiguous()
#         return data

#     # setup dataloaders
#     train_dataloader = TextDataloader(data_prep(train_dataset), max_seq_len)
#     val_dataloader = TextDataloader(data_prep(val_dataset), max_seq_len)
#     test_dataloader = TextDataloader(data_prep(test_dataset), max_seq_len)

#     print(f"[End Load Data] ({time.time() - ts:3f}s)")
#     return train_dataloader, val_dataloader, test_dataloader, vocab, tokenizer

if __name__ == "__main__":
    config = {
        "embedding_dimension": 200,
        "ff_dimension": 200,
        "n_attention_heads": 2,
        "n_encoder_layers": 0,
        "n_decoder_layers": 2,
        "dataset": Dataset.WikiText2.name,
        "segmentation": Segmentation.Subword.name,
        "vocab_size": 40000,
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

    train_dataloader, val_dataloader, test_dataloader, vocab, tokenizer = load_data(config)
    # print(vocab.stoi)
    
    for batch in train_dataloader:
        # print(batch)    
        data, targets = batch
        print("data", data.shape)
        print("targets", targets.shape)
        # break

    print(tokenizer.decode(data[0].tolist()))
    print(tokenizer.decode(targets[0:20].tolist()))

    # check word 
