"""Main training script.

usage: python3 main.py MODEL_NAME
"""

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

import cfg
from asdl.parser import parse as parse_asdl
from data.conala import ConalaDataset
from model.decoder import DecoderLSTM
from model.encoder import EncoderLSTM
from model.seq2seq import Seq2Seq
from utils.tensorboard import profiler, writer


def load(ds):
    """Wrap a Dataset in a DataLoader.

    Variable-length inputs are packed into PackedSequence, which can be fed
    to the seq2seq model directly.
    """

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
    return torch.sum(torch.argmax(logits, dim=2) != label)


def train_epoch(model, ds, optimizer):
    model.train()
    for batch in load(ds):
        optimizer.zero_grad()
        input, label = batch
        logits = model(
            input,
            label,
            max_recipe_len=cfg.max_recipe_len,
            teacher_forcing_p=cfg.teacher_forcing_p,
        )
        loss = calculate_loss(logits, label)
        loss.backward()
        optimizer.step()


def evaluate(model, ds, teacher_forcing_p=0):
    """Evaluate model on a dataset.

    Args:
        model (seq2seq.model.Seq2Seq): seq2seq model to evaluate.
        ds (Dataset): dataset to evaluate on.
        teacher_forcing_p (float): how often to use teacher forcing.

    Returns:
        loss (float): average loss per sample.
        errors (float): average incorrect actions per sample.
    """
    model.eval()
    with torch.no_grad():
        loss = 0
        errors = 0
        for input, label in load(ds):
            logits = model(
                input,
                label,
                max_recipe_len=cfg.max_recipe_len,
                teacher_forcing_p=teacher_forcing_p,
            )
            loss += calculate_loss(logits, label).item()
            errors += calculate_errors(logits, label).item()
    return loss / len(ds), errors / len(ds)


def main():
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SOA]", "[EOA]"]

    # load Python ASDL grammar
    grammar = parse_asdl("src/asdl/Python.asdl")

    # load CoNaLa intent-snippet pairs and map them to tensors
    # FIXME: train_ds and dev_ds are using different word and action mappings.
    # This causes dev performance to become completely nonsense.
    dataset_cache = cfg.model_dir / "dataset_cache.pt"
    if dataset_cache.exists():
        train_ds, dev_ds = torch.load(dataset_cache)
        print("Loaded dataset cache")
    else:
        train_ds = ConalaDataset(
            "data/conala-train.json", grammar=grammar, special_tokens=special_tokens
        )
        dev_ds = ConalaDataset(
            "data/conala-dev.json", grammar=grammar, special_tokens=special_tokens
        )
        torch.save((train_ds, dev_ds), dataset_cache)

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
        writer.add_scalar("Train/loss", train_loss, epoch)
        writer.add_scalar("Train/errors", train_errors, epoch)
        train_tf_loss, train_tf_errors = evaluate(model, train_ds, teacher_forcing_p=1)
        writer.add_scalar("Train/loss(teacher_forcing_p=1)", train_tf_loss, epoch)
        writer.add_scalar("Train/errors(teacher_forcing_p=1)", train_tf_errors, epoch)
        dev_loss, dev_errors = evaluate(model, dev_ds)
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
