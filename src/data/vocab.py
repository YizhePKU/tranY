from collections import Counter


class Vocab:
    """Create a two-way lookup table between words and ids.

    Two special words, <pad> and <unk>, are reserved with id 0 and 1 respectively.

    Args:
        corpus: an iterable of words to include in the vocab.
        freq_cutoff: the minimal number of times a word needs to appear to be included.
        special_words: additional special words to insert into the vocab.
    """

    def __init__(self, corpus, freq_cutoff, special_words=[]):
        self._id2word = ["<pad>", "<unk>"] + special_words
        ctr = Counter(corpus)
        for word, cnt in ctr.most_common():
            if cnt < freq_cutoff:
                break
            self._id2word.append(word)
        self._word2id = {word: id for id, word in enumerate(self._id2word)}

    def id2word(self, id):
        return self._id2word[id]

    def word2id(self, word):
        return self._word2id.get(word, self._word2id["<unk>"])

    def __len__(self):
        return len(self._id2word)
