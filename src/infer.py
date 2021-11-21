from ast import unparse

import torch
from pyrsistent import thaw

from asdl.convert import mr_to_ast
from asdl.parser import parse as parse_asdl
from data.conala import ConalaDataset
from model.seq2seq_v3 import TranY

grammar = parse_asdl("src/asdl/Python.asdl")
special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SOA]", "[EOA]"]

train_ds = ConalaDataset(
    "data/conala-train.json",
    grammar=grammar,
    special_tokens=special_tokens,
    shuffle=False,
)

model = TranY.load_from_checkpoint(
    "tb_logs/TranY/version_7/checkpoints/epoch=281-step=8459.ckpt"
)

while True:
    # example: prepend string str_0 to all items in list str_1
    print()
    intent = input("Prompt: ")
    tokens = intent.split()
    sentence = torch.tensor(
        [train_ds.intent2id.get(token, train_ds.intent2id['[UNK]']) for token in tokens],
    ).unsqueeze(1)
    sentence_length = torch.tensor([len(tokens)])

    results = model.forward_beam_search(
        sentence,
        sentence_length,
        beam_width=15,
        result_count=10,
        action2id=train_ds.action2id,
        id2action=train_ds.id2action,
        grammar=grammar,
    )

    for result in results:
        mr = thaw(result[1])
        pyast = mr_to_ast(mr)
        try:
            snippet = unparse(pyast)
        except Exception as e:
            print("[Bad snippet]")
        print(snippet)
