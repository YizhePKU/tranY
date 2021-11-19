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
        input = torch.nn.utils.rnn.pad_sequence([input for input, _ in data])
        input_lengths = [len(input) for input, _ in data]
        label = torch.nn.utils.rnn.pad_sequence([label for _, label in data])
        label_lengths = [len(label) for _, label in data]
        return (
            input,
            label,
            torch.tensor(input_lengths),
            torch.tensor(label_lengths),
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
trainer = pl.Trainer.from_argparse_args(args, logger=logger)
model = TranY(
    **vars(args),
    encoder_vocab_size=train_ds.intent_vocab_size,
    decoder_vocab_size=train_ds.action_vocab_size
)
trainer.fit(model, train_loader, val_loader)
