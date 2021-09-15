import ast
import random
import torch
import torch.nn as nn
import torch.optim as optim
from asdl.action import mr_to_actions_dfs
from asdl.convert import ast_to_mr

import cfg
import asdl
from data.conala import canonicalize, load_intent_snippet
from data.tokenize import make_lookup_tables, tokenize
from seq2seq.encoder import EncoderLSTM
from seq2seq.decoder import DecoderLSTM
from seq2seq.train import train

PAD = "<PAD>"  # padding
SOS = "<SOS>"  # start-of-sentence
EOS = "<EOS>"  # end-of-sentence
SOA = "<SOA>"  # start-of-actions
EOA = "<EOA>"  # end-of-actions

random.seed(47)
torch.manual_seed(47)

# load Python ASDL grammar
grammar = asdl.parser.parse("src/asdl/Python.asdl")

# load CoNaLa intent-snippet pairs
intent_snippet = load_intent_snippet("data/conala-dev.json")

# convert intent-snippet into canonicalized intent-mr-ph2mr
intent_mr_ph2mr_trippets = []
for intent, snippet in intent_snippet:
    pyast = ast.parse(snippet)
    mr = ast_to_mr(pyast)
    intent_mr_ph2mr_trippets.append(canonicalize(intent, mr))

# convert intent to words, mr to actions
words_actions_pairs = [
    (
        [SOS] + tokenize(intent) + [EOS],
        [SOA] + list(mr_to_actions_dfs(mr, grammar)) + [EOA],
    )
    for intent, mr, _ in intent_mr_ph2mr_trippets
]

# make vocabuary lookup table
idx2word, word2idx = make_lookup_tables(
    [word for words, _ in words_actions_pairs for word in words],
    special=[PAD, SOS, EOS],
)
word_vocab_size = len(idx2word)

# make action lookup table
idx2action, action2idx = make_lookup_tables(
    [action for _, actions in words_actions_pairs for action in actions],
    special=[PAD, SOA, EOA],
)
action_vocab_size = len(idx2action)

# prepare training data
# TODO: trim tails for max_sentence_length and max_action_length
max_sentence_length = max(len(words) for words, _ in words_actions_pairs)
max_action_length = max(len(actions) for _, actions in words_actions_pairs)

# input_tensor: (max_sentence_length x batch_size)
input_tensor = nn.utils.rnn.pad_sequence(
    [
        torch.tensor([word2idx[word] for word in words], device=cfg.device)
        for words, _ in words_actions_pairs
    ]
)

# output_tensor: (max_action_length x batch_size)
output_tensor = nn.utils.rnn.pad_sequence(
    [
        torch.tensor([action2idx[action] for action in actions], device=cfg.device)
        for _, actions in words_actions_pairs
    ]
)

encoder = EncoderLSTM(vocab_size=word_vocab_size, **cfg.EncoderLSTM)
decoder = DecoderLSTM(vocab_size=action_vocab_size, **cfg.DecoderLSTM)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=cfg.learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=cfg.learning_rate)

decoder_init_input = action2idx[SOA]

train(
    encoder=encoder,
    decoder=decoder,
    input_tensor=input_tensor,
    output_tensor=output_tensor,
    encoder_optimizer=encoder_optimizer,
    decoder_optimizer=decoder_optimizer,
    decoder_init_action=decoder_init_input,
    EOA=EOA,
    n_epochs=cfg.n_epochs,
    max_sentence_length=max_action_length,
    max_action_length=max_action_length,
)
