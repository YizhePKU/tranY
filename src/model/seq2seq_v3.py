import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim


class TranY(pl.LightningModule):
    def __init__(
        self,
        encoder_embed_d,
        encoder_hidden_d,
        encoder_nlayers,
        encoder_dropout_p,
        decoder_embed_d,
        decoder_hidden_d,
        decoder_nlayers,
        decoder_dropout_p,
    ):
        self.save_hyperparameters()

    def add_argparse_args(parser):
        parser = parser.add_argument_group("TranY")
        parser.add_argument("--encoder_embed_d", type=int, default=64)
        parser.add_argument("--encoder_hidden_d", type=int, default=128)
        parser.add_argument("--encoder_nlayers", type=int, default=1)
        parser.add_argument("--encoder_dropout_p", type=int, default=0.3)
        parser.add_argument("--decoder_embed_d", type=int, default=64)
        parser.add_argument("--decoder_hidden_d", type=int, default=128)
        parser.add_argument("--decoder_nlayers", type=int, default=1)
        parser.add_argument("--decoder_dropout_p", type=int, default=0.3)
