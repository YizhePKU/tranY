import random
import torch
import torch.optim as optim
import torch.nn.functional as F

import cfg
from utils.tensorboard import writer, profiler
from utils.checkpoints import Checkpoints
from asdl.parser import parse as parse_asdl
from data.conala import ConalaDataset
from seq2seq.encoder import EncoderLSTM
from seq2seq.decoder import DecoderLSTM
from seq2seq.model import Seq2Seq


def load(ds):
    def collate_fn(data):
        return (
            torch.nn.utils.rnn.pack_sequence(
                [input for input, _ in data], enforce_sorted=False
            ).to(cfg.device),
            torch.nn.utils.rnn.pack_sequence(
                [label for _, label in data], enforce_sorted=False
            ).to(cfg.device),
        )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
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
        logits, _ = model(input.to(cfg.device), max_action_len=cfg.max_action_len)
        loss = calculate_loss(logits, label.to(cfg.device))
        loss.backward()
        optimizer.step()


def evaluate(model, ds):
    """Evaluate model on a dataset.

    Args:
        model (seq2seq.model.Seq2Seq): seq2seq model to evaluate.
        ds (Dataset): dataset to evaluate on.

    Returns:
        loss (float): average loss per sample.
        errors (float): average incorrect actions per sample.
    """
    model.eval()
    with torch.no_grad():
        loss = 0
        errors = 0
        for input, label in load(ds):
            logits, _ = model(input, max_action_len=cfg.max_action_len)
            loss += calculate_loss(logits, label.to(cfg.device)).item()
            errors += calculate_errors(logits, label).item()
    return loss / len(ds), errors / len(ds)


def main():
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

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
    encoder = EncoderLSTM(
        vocab_size=train_ds.word_vocab_size,
        device=cfg.device,
        **cfg.EncoderLSTM,
    )
    decoder = DecoderLSTM(
        vocab_size=train_ds.action_vocab_size,
        device=cfg.device,
        **cfg.DecoderLSTM,
    )
    model = Seq2Seq(encoder, decoder, special_tokens=special_tokens, device=cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # load checkpoints
    if pt := cfg.checkpoints.latest():
        state = torch.load(pt)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        epoch = state["epoch"]
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        epoch = 0

    # start training loop
    while True:
        epoch += 1
        print(f"Epoch {epoch}")
        train_epoch(model, train_ds, optimizer)

        train_loss, train_errors = evaluate(model, train_ds)
        dev_loss, dev_errors = evaluate(model, dev_ds)
        writer.add_scalar("Train/loss", train_loss, epoch)
        writer.add_scalar("Train/errors", train_errors, epoch)
        writer.add_scalar("Dev/loss", dev_loss, epoch)
        writer.add_scalar("Dev/errors", dev_errors, epoch)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, cfg.checkpoints.new())


if __name__ == "__main__":
    main()
