from argparse import ArgumentParser
import ast
import re

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from pyrsistent import thaw

from asdl.convert import mr_to_ast
from asdl.parser import parse as parse_asdl
from data.conala_v2 import ConalaDataset, unprocess_snippet
from model.seq2seq_v3 import TranY

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

# TODO: DRY
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

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=47)
parser.add_argument("--batch_size", type=int, default=64)
parser = TranY.add_argparse_args(parser)
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

dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=64, collate_fn=collate_fn)
model = TranY.load_from_checkpoint(
    "tb_logs/TranY/post_conala_v2_early_stop/checkpoints/epoch=38-step=1169.ckpt"
)
model.eval()

# TODO: cleanup
scores = []
for idx in range(len(dev_ds)):
    intent1, snippet1, slot = dev_ds.intent_snippet_slots[idx]
    snippet = unprocess_snippet(snippet1, slot)
    print()
    print(f"intent1: {intent1}")
    print(f"snippet1: {snippet1}")
    print(f"snippet: {snippet}")
    print(f"slot: {slot}")

    sentence = dev_ds.sentences[idx].unsqueeze(1)
    sentence_length = torch.tensor([len(sentence)])

    results = model.forward_beam_search(
        sentence,
        sentence_length,
        beam_width=15,
        result_count=1,
        action_vocab=train_ds.action_vocab,
        grammar=parse_asdl("src/asdl/Python.asdl"),
    )
    if results:
        mr = thaw(results[0][1])
        pyast = mr_to_ast(mr)
        infered_snippet = ast.unparse(pyast)
        print(f"infered_snippet: {infered_snippet}")
        unprocessed_infered_snippet = unprocess_snippet(infered_snippet, slot)
        print(f"unprocess_infered_snippet: {unprocessed_infered_snippet}")

        bleu = snippet_bleu(unprocessed_infered_snippet, snippet)
        print(f"bleu: {bleu}")
    else:
        print("Failed to infer any snippet")
        bleu = 0

    scores.append(bleu)
    print(f"Running average: {sum(scores) / len(scores)}, {len(scores)} samples")
