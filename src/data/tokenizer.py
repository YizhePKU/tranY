from collections import Counter
from itertools import chain


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
        raise ValueError("vocabulary is readonly")

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return "Vocabulary[size=%d]" % len(self)

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
        # print(
        #     "number of word types: %d, number of word types w/ frequency > 1: %d"
        #     % (len(word_freq), len(non_singletons))
        # )
        # print("singletons: %s" % singletons)

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]
        words_not_included = []
        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)
                else:
                    words_not_included.append(word)
        # print("word types not included: %d" % len(words_not_included))

        return vocab_entry
