import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger

import cfg
from asdl.parser import parse as parse_asdl
from data.conala import ConalaDataset
from model.seq2seq_v2 import DecoderLSTM, EncoderDecoder, EncoderLSTM

torch.manual_seed(cfg.seed)


def load(ds, shuffle=False):
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
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def calculate_loss(logits, label):
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), label.flatten(), reduction="sum"
    )


def calculate_errors(logits, label):
    return torch.sum(torch.argmax(logits, dim=2) != label)


grammar = parse_asdl("src/asdl/Python.asdl")
special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SOA]", "[EOA]"]

train_ds = ConalaDataset(
    "data/conala-train.json", grammar=grammar, special_tokens=special_tokens
)
val_ds = ConalaDataset(
    "data/conala-dev.json",
    grammar=grammar,
    special_tokens=special_tokens,
    action_vocab=train_ds.action_vocab,
    intent_vocab=train_ds.intent_vocab,
    shuffle=False,
)
train_loader = load(train_ds)
val_loader = load(val_ds)


class TranY(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = EncoderLSTM(
            vocab_size=train_ds.intent_vocab_size,
            **cfg.EncoderLSTM,
        )
        self.decoder = DecoderLSTM(
            action_vocab_size=train_ds.action_vocab_size,
            **cfg.DecoderLSTM,
        )
        self.model = EncoderDecoder(self.encoder, self.decoder)

    def forward(self, input, label, input_length, label_length):
        return self.model(input, label, input_length, label_length)

    def training_step(self, batch, batch_idx):
        input, label, input_length, label_length = batch
        input = input.to(self.device)
        label = label.to(self.device)
        input_length = input_length.to(self.device)
        label_length = label_length.to(self.device)
        logits = self(input, label, input_length, label_length)
        loss = calculate_loss(logits, label)
        errors = calculate_errors(logits, label)
        return {
            "loss": loss,
            "errors": errors,
        }

    def training_epoch_end(self, outputs):
        loss = [d["loss"] for d in outputs]
        avg_loss = sum(loss) / len(loss)
        errors = [d["errors"] for d in outputs]
        avg_errors = sum(errors) / len(errors)
        self.log_dict(
            {
                "Train/loss": avg_loss,
                "Train/errors": avg_errors,
            }
        )

    def validation_step(self, batch, batch_idx):
        input, label, input_length, label_length = batch
        input = input.to(self.device)
        label = label.to(self.device)
        input_length = input_length.to(self.device)
        label_length = label_length.to(self.device)
        logits = self(input, label, input_length, label_length)
        loss = calculate_loss(logits, label)
        errors = calculate_errors(logits, label)
        return {
            "loss": loss,
            "errors": errors,
        }

    def validation_epoch_end(self, outputs):
        loss = [d["loss"] for d in outputs]
        avg_loss = sum(loss) / len(loss)
        errors = [d["errors"] for d in outputs]
        avg_errors = sum(errors) / len(errors)
        self.log_dict(
            {
                "Train/loss": avg_loss,
                "Train/errors": avg_errors,
            }
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=cfg.learning_rate)


logger = TensorBoardLogger("tb_logs", name="TranY")
trainer = pl.Trainer(gpus=1, logger=logger)
model = TranY()
trainer.fit(model, train_loader, val_loader)
