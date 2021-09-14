def tokenize(intent):
    """Tokenize (canonicalized) intent into a sequence of words.

    Args:
        intent (str): canonicalized intent.

    Returns:
        (list[str]): sequence of words.
    """
    return intent.split()


def make_lookup_tables(vocab):
    """Create word2idx/idx2word tables for a given vocabuary.

    Args:
        vocab (Iterable[str]): vocabulary to create lookup tables for.

    Returns:
        (list[str], dict[str,int]): idx2word, word2idx
            mappings between a word and its integer index for the vocabuary.
    """
    idx2word = list(set(vocab))
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    return idx2word, word2idx
