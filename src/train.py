from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from asdl.parser import parse as parse_asdl
from data.conala import ConalaDataset
from model.seq2seq_v3 import TranY


def load(ds, batch_size, shuffle=False):
    """Wrap a Dataset in a DataLoader."""

    def collate_fn(data):
        sentence = torch.nn.utils.rnn.pad_sequence([input for input, _ in data])
        sentence_length = [len(input) for input, _ in data]
        label = torch.nn.utils.rnn.pad_sequence([label for _, label in data])
        label_length = [len(label) for _, label in data]
        return (
            sentence,
            label,
            torch.tensor(sentence_length),
            torch.tensor(label_length),
        )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=47)
parser.add_argument("--batch_size", type=int, default=16)
parser = TranY.add_argparse_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()
torch.manual_seed(args.seed)

grammar = parse_asdl("src/asdl/Python.asdl")
special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SOA]", "[EOA]"]

train_ds = ConalaDataset(
    "data/conala-train.json",
    grammar=grammar,
    special_tokens=special_tokens,
    shuffle=False,
)
val_ds = ConalaDataset(
    "data/conala-dev.json",
    grammar=grammar,
    special_tokens=special_tokens,
    action_vocab=train_ds.action_vocab,
    intent_vocab=train_ds.intent_vocab,
    shuffle=False,
)
train_loader = load(train_ds, args.batch_size)
val_loader = load(val_ds, args.batch_size)

logger = TensorBoardLogger("tb_logs", name="TranY")
trainer = pl.Trainer.from_argparse_args(args, logger=logger, profiler="simple")
# model = TranY(
#     **vars(args),
#     encoder_vocab_size=train_ds.intent_vocab_size,
#     decoder_vocab_size=train_ds.action_vocab_size
# )
# trainer.fit(model, train_loader, val_loader)

model = TranY.load_from_checkpoint('tb_logs/TranY/version_2/checkpoints/epoch=21-step=2617.ckpt')
sentence = torch.tensor(
    [val_ds.intent2id[token] for token in "sort array in descending order".split()],
).unsqueeze(1)
sentence_length = torch.tensor([5])
results = model.forward_beam_search(
    sentence,
    sentence_length,
    beam_width=3,
    result_count=5,
    action2id=val_ds.action2id,
    id2action=val_ds.id2action,
    grammar=grammar,
)
from pyrsistent import thaw
from asdl.convert import mr_to_ast
from ast import parse, unparse