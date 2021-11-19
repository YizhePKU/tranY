from argparse import ArgumentParser
import pytorch_lightning as pl
from model.seq2seq_v3 import TranY

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=47)
parser = TranY.add_argparse_args(parser)
parser = pl.Trainer.add_argparse_args(parser)