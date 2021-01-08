
# imports
import enum

from torch.nn import Transformer
import io
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# constants/enums
class Dataset(enum.Enum):
    PennTreebank = 0,
    WikiText2 = 1,
    WikiText103 = 2


class LanguageTask(enum.Enum):
    CausalLanuageModeling = 0,
    MaskedLanuageModeling = 1


class Segmentation(enum.Enum):
    Word = 0,
    Subword = 1
    Character = 2
    BPE = 3
    BBPE = 4
    BYTE = 5

# load training data
# change to use torchtext
def load_data():
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    test_filepath, valid_filepath, train_filepath = extract_archive(
        download_from_url(url))
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer,
                                          iter(io.open(train_filepath,
                                                       encoding="utf8"))))  # should build from all text

    def data_process(raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                             dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(iter(io.open(train_filepath, encoding="utf8")))
    val_data = data_process(iter(io.open(valid_filepath, encoding="utf8")))
    test_data = data_process(iter(io.open(test_filepath, encoding="utf8")))

    return train_data, val_data, test_data, vocab

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

# utils
def extract_config(config, *argv, ):
    assert len(argv) > 0, "No keys to extract"
    config_values = []
    for key in argv:
        assert key in config, f"Key '{key}' not in config"
        config_values.append(config[key])

    return tuple(config_values) if len(argv) > 1 else config_values[0]


if __name__ == "__main__":
    # configure model
    config = {
        "embedding_dimension": 200,
        "ff_dimension": 200,
        "n_attention_heads": 2,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "dataset": Dataset.PennTreebank,
        "max_seq_len": 35,
        "batch_size": 20,
        "eval_batch_size": 10,
    }

    # configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load training data
    train_data, val_data, test_data, vocab = load_data()

    # batch data
    train_data_batches = batchify(train_data, config["batch_size"], device)
    val_data_batches = batchify(val_data, config["eval_batch_size"], device)
    test_data_batches = batchify(test_data, config["eval_batch_size"], device)


    # instantiate model
    dataset, n_attention_heads = extract_config(config)

    model = Transformer()

    # training loop

    # evaluation
