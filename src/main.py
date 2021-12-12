import argparse
import ast
import re
from pathlib import Path

import pytorch_lightning as pl
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from asdl.convert import mr_to_ast

from asdl.parser import parse as parse_asdl
from data.conala_v2 import ConalaDataset, unprocess_snippet
from model.seq2seq_v3 import TranY
from utils.arggroup import patch_arggroup


def tokenize_for_bleu_eval(snippet):
    """Tokenize snippet for BLEU evaluation.

    Taken from Wang Ling et al., Latent Predictor Networks for Code Generation (2016).
    """
    snippet = re.sub(r"([^A-Za-z0-9_])", r" \1 ", snippet)
    snippet = re.sub(r"([a-z])([A-Z])", r"\1 \2", snippet)
    snippet = re.sub(r"\s+", " ", snippet)
    snippet = snippet.replace('"', "`")
    snippet = snippet.replace("'", "`")
    tokens = [t for t in snippet.split(" ") if t]
    return tokens


def snippet_bleu(lhs_snippet, rhs_snippet):
    return sentence_bleu(
        [tokenize_for_bleu_eval(lhs_snippet)],
        tokenize_for_bleu_eval(rhs_snippet),
        smoothing_function=SmoothingFunction().method3,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate", "infer"])
    parser.add_argument("model_name")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--no_use_simplified_grammar",
        dest="use_simplified_grammar",
        action="store_false",
    )
    parser.set_defaults(use_simplified_grammar=True)
    parser = TranY.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ConalaDataset.add_argparse_args(parser)

    args = parser.parse_args()
    patch_arggroup(args)
    return args


args = parse_args()
torch.manual_seed(args.seed)

if args.use_simplified_grammar:
    grammar = parse_asdl("src/asdl/python3_simplified.asdl")
else:
    grammar = parse_asdl("src/asdl/python3.asdl")

train_ds = ConalaDataset("data/conala-train.json", grammar, **args.group(ConalaDataset))
val_ds = ConalaDataset(
    "data/conala-val.json",
    grammar,
    intent_vocab=train_ds.intent_vocab,
    action_vocab=train_ds.action_vocab,
    **args.group(ConalaDataset),
)


def collate_fn(data):
    sentence_tensor = torch.nn.utils.rnn.pad_sequence(
        [sentence for sentence, _ in data]
    )
    sentence_length = [len(sentence) for sentence, _ in data]
    recipe_tensor = torch.nn.utils.rnn.pad_sequence([recipe for _, recipe in data])
    recipe_length = [len(recipe) for _, recipe in data]
    return (
        sentence_tensor,
        recipe_tensor,
        torch.tensor(sentence_length),
        torch.tensor(recipe_length),
    )


train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=args.batch_size, collate_fn=collate_fn
)
val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size=args.batch_size, collate_fn=collate_fn
)

if args.mode == "train":
    logger = TensorBoardLogger("tb_logs", name=args.model_name)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        profiler="simple",
        callbacks=[EarlyStopping(monitor="Val/loss")],
    )
    model = TranY(
        encoder_vocab_size=len(train_ds.intent_vocab),
        decoder_vocab_size=len(train_ds.action_vocab),
        **args.group(TranY),
    )
    trainer.fit(model, train_dl, val_dl)
elif args.mode == "evaluate":
    model_dir = Path("tb_logs") / args.model_name
    checkpoint_dir = max(model_dir.iterdir()) / "checkpoints"
    checkpoint = next(checkpoint_dir.iterdir())

    model = TranY.load_from_checkpoint(checkpoint)
    model.eval()

    # TODO: make evaluation parallel
    scores = []
    ds = val_ds
    for idx in range(len(ds)):
        intent1, snippet1, slot = ds.intent_snippet_slots[idx]
        snippet = unprocess_snippet(snippet1, slot)
        print()
        print(f"intent1: {intent1}")
        print(f"snippet1: {snippet1}")
        print(f"snippet: {snippet}")
        print(f"slot: {slot}")

        sentence = ds.sentences[idx].unsqueeze(1)
        sentence_length = torch.tensor([len(sentence)])

        results = model.forward_beam_search(
            sentence,
            sentence_length,
            beam_width=15,
            result_count=10,
            action_vocab=train_ds.action_vocab,
            grammar=grammar,
        )
        if results:
            mr = results[0][1]
            pyast = mr_to_ast(mr)
            infered_snippet = ast.unparse(pyast)
            print(f"infered_snippet: {infered_snippet}")
            unprocessed_infered_snippet = unprocess_snippet(infered_snippet, slot)
            print(f"unprocess_infered_snippet: {unprocessed_infered_snippet}")
            bleu = snippet_bleu(unprocessed_infered_snippet, snippet)
            print(f"bleu: {bleu:.4f}")
        else:
            print("Failed to infer any snippet")
            bleu = 0

        scores.append(bleu)
        print(f"Running average: {sum(scores) / len(scores):.4f}, {len(scores)} samples")
elif args.mode == "infer":
    while True:
        model_dir = Path("tb_logs") / args.model_name
        checkpoint_dir = max(model_dir.iterdir()) / "checkpoints"
        checkpoint = next(checkpoint_dir.iterdir())

        model = TranY.load_from_checkpoint(checkpoint)
        model.eval()

        # example: prepend string str_0 to all items in list str_1
        print()
        intent = input("Prompt: ")
        tokens = intent.split()
        sentence = torch.tensor(
            [
                train_ds.intent_vocab.word2id(token)
                for token in tokens
            ],
        ).unsqueeze(1)
        sentence_length = torch.tensor([len(tokens)])

        results = model.forward_beam_search(
            sentence,
            sentence_length,
            beam_width=15,
            result_count=10,
            action_vocab=train_ds.action_vocab,
            grammar=grammar,
        )

        for result in results:
            mr = result[1]
            pyast = mr_to_ast(mr)
            try:
                snippet = ast.unparse(pyast)
            except Exception as e:
                print("[Bad snippet]")
            print(snippet)

