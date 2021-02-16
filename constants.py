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
