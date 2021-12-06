from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data.conala_v2 import ConalaDataset
from model.seq2seq_v3 import TranY

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=47)
parser.add_argument("--batch_size", type=int, default=64)
parser = TranY.add_argparse_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
parser = ConalaDataset.add_argparse_args(parser)
args = parser.parse_args()
torch.manual_seed(args.seed)

train_ds = ConalaDataset("data/conala-train.json", **vars(args))
dev_ds = ConalaDataset(
    "data/conala-dev.json",
    intent_vocab=train_ds.intent_vocab,
    action_vocab=train_ds.action_vocab,
    **vars(args),
)

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

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn)
dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate_fn)

logger = TensorBoardLogger("tb_logs", name="TranY")
trainer = pl.Trainer.from_argparse_args(
    args,
    # logger=logger,
    profiler="simple",
    callbacks=[EarlyStopping(monitor="Val/loss")],
)
model = TranY(
    **vars(args),
    encoder_vocab_size=len(train_ds.intent_vocab),
    decoder_vocab_size=len(train_ds.action_vocab),
)
trainer.fit(model, train_dl, dev_dl)
