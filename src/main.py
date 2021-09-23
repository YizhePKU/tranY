import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

import cfg
from asdl.parser import parse as parse_asdl
from data.conala import ConalaDataset
from seq2seq.encoder import EncoderLSTM
from seq2seq.decoder import DecoderLSTM
from seq2seq.model import Seq2Seq

random.seed(47)
torch.manual_seed(47)

special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SOA]", "[EOA]"]

# load Python ASDL grammar
grammar = parse_asdl("src/asdl/Python.asdl")

# load CoNaLa intent-snippet pairs and map them to tensors
train_ds = ConalaDataset(
    "data/conala-train.json", grammar=grammar, special_tokens=special_tokens
)
dev_ds = ConalaDataset(
    "data/conala-dev.json", grammar=grammar, special_tokens=special_tokens
)

# initialize the model and optimizer
encoder = EncoderLSTM(vocab_size=train_ds.word_vocab_size, **cfg.EncoderLSTM)
decoder = DecoderLSTM(vocab_size=train_ds.action_vocab_size, **cfg.DecoderLSTM)
model = Seq2Seq(encoder, decoder, special_tokens=special_tokens, device=cfg.device)
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)


def load(ds, batch_size=cfg.batch_size):
    def collate_fn(data):
        return (
            torch.nn.utils.rnn.pack_sequence(
                [input for input, _ in data], enforce_sorted=False
            ),
            torch.nn.utils.rnn.pack_sequence(
                [label for _, label in data], enforce_sorted=False
            ),
        )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def calculate_loss(logits, label):
    assert type(label) == torch.nn.utils.rnn.PackedSequence
    label, _ = torch.nn.utils.rnn.pad_packed_sequence(label)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), label.flatten(), reduction="sum"
    )


def calculate_errors(logits, label):
    assert type(label) == torch.nn.utils.rnn.PackedSequence
    label, _ = torch.nn.utils.rnn.pad_packed_sequence(label)
    return torch.sum(torch.argmax(logits, dim=2) != label.to(cfg.device))


def train_epoch(model, ds, optimizer):
    model.train()
    for batch in load(ds):
        optimizer.zero_grad()
        input, label = batch
        logits, actions = model(input.to(cfg.device), max_action_len=cfg.max_action_len)
        loss = calculate_loss(logits, label.to(cfg.device))
        loss.backward()
        optimizer.step()


def evaluate(model, ds):
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch in load(ds):
            input, label = batch
            logits, actions = model(
                input.to(cfg.device), max_action_len=cfg.max_action_len
            )
            assert torch.all(torch.argmax(logits, dim=2) == actions)
            loss += calculate_loss(logits, label.to(cfg.device)).item()
    return loss


if __name__ == "__main__":
    for epoch in range(cfg.n_epochs):
        print("EpochStart", {"epoch": epoch})
        train_epoch(model, train_ds, optimizer)
        print("EpochEnd", {"epoch": epoch})

        train_loss = evaluate(model, train_ds)
        dev_loss = evaluate(model, dev_ds)
        print("Evaluate", {"train_loss": train_loss, "dev_loss": dev_loss})

    # save after training
    saved_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "dev_loss": dev_loss,
    }
    path = Path("models/model.pt")
    path.parent.mkdir(exist_ok=True)
    torch.save(saved_data, path)
    print("SaveModel", saved_data)
