from argparse import ArgumentParser
from ast import unparse

import torch
from pyrsistent import thaw

from asdl.convert import mr_to_ast
from asdl.parser import parse as parse_asdl
from data.conala_v2 import ConalaDataset
from model.seq2seq_v3 import TranY

torch.manual_seed(0)

# TODO: DRY
def collate_fn(data):
    sentence_tensor = torch.nn.utils.rnn.pad_sequence([sentence for sentence, _ in data])
    sentence_length = [len(sentence) for sentence, _ in data]
    recipe_tensor = torch.nn.utils.rnn.pad_sequence([recipe for _, recipe in data])
    recipe_length = [len(recipe) for _, recipe in data]
    return (
        sentence_tensor,
        recipe_tensor,
        torch.tensor(sentence_length),
        torch.tensor(recipe_length),
    )

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=47)
parser.add_argument("--batch_size", type=int, default=64)
parser = TranY.add_argparse_args(parser)
parser = ConalaDataset.add_argparse_args(parser)
args = parser.parse_args()
torch.manual_seed(args.seed)

grammar = parse_asdl("src/asdl/python3.asdl")

train_ds = ConalaDataset("data/conala-train.json", grammar, **vars(args))
dev_ds = ConalaDataset(
    "data/conala-dev.json",
    grammar,
    intent_vocab=train_ds.intent_vocab,
    action_vocab=train_ds.action_vocab,
    **vars(args),
)

dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=64, collate_fn=collate_fn)
model = TranY.load_from_checkpoint(
    "tb_logs/TranY/post_conala_v2_early_stop/checkpoints/epoch=38-step=1169.ckpt"
)

while True:
    # example: prepend string str_0 to all items in list str_1
    print()
    intent = input("Prompt: ")
    tokens = intent.split()
    sentence = torch.tensor(
        [
            train_ds.intent_vocab.word2id(token)
            for token in tokens
        ],
    ).unsqueeze(1)
    sentence_length = torch.tensor([len(tokens)])

    results = model.forward_beam_search(
        sentence,
        sentence_length,
        beam_width=15,
        result_count=10,
        action_vocab=train_ds.action_vocab,
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
