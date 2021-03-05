import enum

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


DATA_PATH = './.data/'
TRAINING_DATA = {
    'PennTreebank': {
        "location": "penn-treebank/",
        "filenames": ['ptb.train.tokens', 'ptb.valid.tokens', 'ptb.test.tokens']
    },
    'WikiText2': {
        "location": "wikitext-2/wikitext-2/",
        "filenames": ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']
    },
    'WikiText103': {
        "location": "wikitext-103/wikitext-103/",
        "filenames": ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']
    }
}

# global variables
# WANDB_ENTITY = "skgbafa"
WANDB_ENTITY = "openai-scholars"
