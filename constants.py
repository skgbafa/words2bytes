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
    BBPE = 3
    BYTE = 4


DATA_PATH = './.data/'
TRAINING_DATA = {
    # 'PennTreebank': {
    #     "location": "penn-treebank-raw/",
    #     "filenames": list(map(lambda x: "wsj_" + format(x, '04'), range(1, 200)))
    # },
    'PennTreebank': {
        "location": "penn-treebank/",
        "filenames": ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']
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

DEBUG_ON=False
