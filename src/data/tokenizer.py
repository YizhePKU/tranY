from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def train_tokenizer(text):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ("[SOS]", 1),
            ("[EOS]", 2),
        ],
    )
    trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"])
    tokenizer.train_from_iterator(text, trainer=trainer)
    return tokenizer


def make_lookup_tables(vocab, special_tokens=[]):
    """Create word2id/id2word tables for a given vocabuary.

    Args:
        vocab (list[str]): vocabulary to create lookup tables for.
        special (list[str]): special tokens that should be placed first at idx2word.

    Returns:
        (list[str], dict[str,int]): idx2word, word2idx
            mappings between a word and its integer index for the vocabuary.
    """
    id2word = special_tokens + list(set(vocab) - set(special_tokens))
    word2id = {word: idx for idx, word in enumerate(id2word)}
    return id2word, word2id
