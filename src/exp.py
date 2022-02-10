import math
import pdb

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.nn.utils.rnn import pad_sequence

from asdl.parser import parse as parse_asdl
from data.conala_v2 import ConalaDataset

grammar = parse_asdl("src/asdl/python3_simplified.asdl")
train_ds = ConalaDataset(
    "data/conala-train.json",
    grammar,
    max_sentence_len=80,
    max_recipe_len=80,
    intent_freq_cutoff=3,
    action_freq_cutoff=3,
)
val_ds = ConalaDataset(
    "data/conala-val.json",
    grammar,
    max_sentence_len=80,
    max_recipe_len=80,
    intent_freq_cutoff=3,
    action_freq_cutoff=3,
    intent_vocab=train_ds.intent_vocab,
    action_vocab=train_ds.action_vocab,
)


def collate_fn(data):
    sentence_tensor = pad_sequence([sentence for sentence, _ in data])
    recipe_tensor = pad_sequence([recipe for _, recipe in data])
    return (
        sentence_tensor,
        recipe_tensor,
    )


train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64, collate_fn=collate_fn)

# Copied from https://pytorch.org/tutorials/beginner/translation_transformer.html
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(pl.LightningModule):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
        )
        return self.generator(outs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src,
            tgt_input,
            device=self.device,
        )
        logits = self.forward(
            src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
        )
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
        )
        error_rate = (
            torch.sum(torch.argmax(logits, dim=2) != tgt_output) / tgt_output.numel()
        )
        return {
            "loss": loss,
            "error_rate": error_rate,
        }

    def validation_step(self, batch, batch_idx):
        # same as training_step for now
        src, tgt = batch
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src,
            tgt_input,
            device=self.device,
        )
        logits = self.forward(
            src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
        )
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
        )
        error_rate = (
            torch.sum(torch.argmax(logits, dim=2) != tgt_output) / tgt_output.numel()
        )
        return {
            "loss": loss,
            "error_rate": error_rate,
        }

    def _on_epoch_end(self, outputs, group):
        with torch.no_grad():
            losses = [output["loss"] for output in outputs]
            avg_loss = sum(losses) / len(losses)
            error_rates = [output["error_rate"] for output in outputs]
            avg_error_rate = sum(error_rates) / len(error_rates)
            self.log(f"{group}/loss", avg_loss)
            self.log(f"{group}/error_rate", avg_error_rate)

    def training_epoch_end(self, outputs):
        self._on_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._on_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.test_epoch_end(outputs, "test")


def create_mask(src, tgt, device="cpu"):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len))

    # `nn.Transformer` expects `src_padding_mask` and `tgt_padding_mask` to have batch dimension first.
    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return (
        src_mask.to(device),
        tgt_mask.to(device),
        src_padding_mask.to(device),
        tgt_padding_mask.to(device),
    )


model = Seq2SeqTransformer(
    num_encoder_layers=6,
    num_decoder_layers=6,
    emb_size=256,
    nhead=8,
    src_vocab_size=len(train_ds.intent_vocab),
    tgt_vocab_size=len(train_ds.action_vocab),
    dim_feedforward=1024,
    dropout=0.1,
)
trainer = pl.Trainer(
    gpus=[0],
    logger=WandbLogger(project="tranY"),
    callbacks=[EarlyStopping(monitor="val/loss")],
)
trainer.fit(model, train_dl, val_dl)
