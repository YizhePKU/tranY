#!/usr/env/bin python3
import ast
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import asdl.parser
from data.conala import load_intent_snippet, canonicalize, uncanonicalize
from data.tokenize import make_lookup_tables, tokenize
from asdl.convert import ast_to_mr, mr_to_ast
from asdl.action import extract_cardinality, mr_to_actions_dfs, actions_to_mr_dfs
from seq2seq.encoder import EncoderLSTM
from seq2seq.decoder import DecoderLSTM

SOS = "<SOS>"
EOS = "<EOS>"

random.seed(47)
torch.manual_seed(47)

device = torch.device("cpu")

# load Python ASDL grammar
grammar = asdl.parser.parse("src/asdl/Python.asdl")
cardinality = extract_cardinality(grammar)

# load CoNaLa intent-snippet pairs
intent_snippets = load_intent_snippet("data/conala-dev.json")

# convert intent-snippet into canonicalized intent-mr-ph2mr
intent_mr_ph2mr = []
for intent, snippet in intent_snippets:
    pyast = ast.parse(snippet)
    mr = ast_to_mr(pyast)
    intent_mr_ph2mr.append(canonicalize(intent, mr))

# make vocabuary lookup table
idx2word, word2idx = make_lookup_tables(
    [word for intent, _, _ in intent_mr_ph2mr for word in tokenize(intent)] + [EOS]
)

# make action lookup table
idx2action, action2idx = make_lookup_tables(
    [
        action
        for _, mr, _ in intent_mr_ph2mr
        for action in mr_to_actions_dfs(mr, grammar)
    ]
    + [SOS, EOS]
)

# make input-output tensors
# append EOS to both input and output
def make_tensor_pair(intent, mr):
    input_words = tokenize(intent) + [EOS]
    input_tensor = torch.tensor([word2idx[word] for word in input_words], device=device)
    output_words = list(mr_to_actions_dfs(mr, grammar)) + [EOS]
    output_tensor = torch.tensor(
        [action2idx[action] for action in output_words], device=device
    )
    return input_tensor, output_tensor


def train(encoder, decoder, n_iters, learning_rate, max_action_length):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [
        make_tensor_pair(*random.choice(intent_mr_ph2mr)[:2]) for i in range(n_iters)
    ]
    for i in range(n_iters):
        print(f"Iteration {i+1}/{n_iters}")
        input_tensor, output_tensor = training_pairs[i]
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # insert a batch dimension into input_tensor
        # input_tensor: (input_length x 1)
        input_tensor = input_tensor.unsqueeze(1)

        # feed input_tensor to encoder
        # encoder_outputs: (input_length x 1 x hidden_size)
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # generate model-predicted output one at a time
        # initialize decoder_hidden with the final value of encoder_hidden
        decoder_logits = []
        decoder_hidden = encoder_hidden
        last_output = torch.tensor(action2idx[SOS])
        while last_output != action2idx[EOS] and len(decoder_logits) < max_action_length:
            # reshape last_output to (1, 1) because seq_length == batch_size == 1
            # logits: (1 x vocab_size)
            logits, decoder_hidden = decoder(
                last_output.view((1, 1)), decoder_hidden
            )
            # TODO: is it OK to sample with argmax?
            last_output = torch.argmax(logits)
            decoder_logits.append(logits)

        # calculate loss
        loss = 0
        for i in range(len(output_tensor)):
            loss += F.cross_entropy(decoder_logits[i], output_tensor[i].unsqueeze(0))

        print(f"    loss: {loss}")

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()


config = {
    "EncoderLSTM": {
        "vocab_size": len(idx2word),
        "embedding_dim": 64,
        "hidden_size": 256,
        "device": device,
    },
    "DecoderLSTM": {
        "vocab_size": len(idx2action),
        "embedding_dim": 128,
        "hidden_size": 256,
        "device": device,
    },
    "train": {
        "n_iters": 10000,
        "learning_rate": 1e-3,
        "max_action_length": 100,
    },
}

encoder = EncoderLSTM(**config["EncoderLSTM"])
decoder = DecoderLSTM(**config["DecoderLSTM"])

train(encoder, decoder, **config["train"])
