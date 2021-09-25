from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from collections import Counter
from itertools import chain


def train_intent_tokenizer(intents, special_tokens=[]):
    """Train a tokenizer for intents.

    Args:
        intents (list[str]): intents to extract vocabulary from.

    Returns:
        (tokenizers.models.Model): a tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ("[SOS]", 2),
            ("[EOS]", 3),
        ],
    )
    trainer = BpeTrainer(special_tokens=special_tokens)
    tokenizer.train_from_iterator(intents, trainer=trainer)
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



class Vocab(object):
    def __init__(self, special_tokens):
        self.word2id = dict()
        for token in special_tokens:
            if token not in self.word2id:
                le = len(self.word2id)
                self.word2id[token] = le
        assert "[UNK]" in special_tokens, "Please add `[UNK]` in special_tokens."
        assert "[PAD]" in special_tokens, "Please add `[PAD]` in special_tokens."
        self.unk_id = self.word2id["[UNK]"]
        self.pad_id = self.word2id["[PAD]"]
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def is_unk(self, word):
        return word not in self

    def to_dict(self):
        return self.word2id

    @staticmethod
    def from_corpus(special_tokens, corpus, size, freq_cutoff=0):
        vocab_entry = Vocab(special_tokens)
        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        singletons = [w for w in word_freq if word_freq[w] == 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]
        words_not_included = []
        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)
                else:
                    words_not_included.append(word)
        print('word types not included: %d' % len(words_not_included))

        return vocab_entry