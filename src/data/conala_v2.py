import ast
import json
import re

from nltk.tokenize import word_tokenize
import torch
from asdl.convert import ast_to_mr
from asdl.recipe import mr_to_recipe_dfs

from data.vocab import Vocab


def load_conala_from_json(filepath):
    """Load the CoNaLa dataset from a JSON file.

    Returns a list of (intent, snippet) pairs.
    """
    with open(filepath) as file:
        data = json.load(file)

    pairs = []
    for sample in data:
        if sample["rewritten_intent"] is None:
            intent = sample["intent"]
        else:
            intent = sample["rewritten_intent"]
        snippet = sample["snippet"]
        pairs.append((intent, snippet))
    return pairs


def process_intent(intent):
    """Extract quoted words and replace them with placeholders.

    Returns:
        intent: the processed intent
        slots: a dict from placeholder strings to (value, quote_mark) pairs,
            where value is the string this placeholder replaces and quote_mark is one of ", ', `
    """
    matches = re.findall(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)", intent)
    var_id = 0
    str_id = 0
    slots = {}
    for match in matches:
        quote_mark, value = match
        # infer whether the quoted value is an identifier
        if quote_mark == "`" and value.isidentifier():
            ph = f"var_{var_id}"
            var_id += 1
        else:
            ph = f"str_{str_id}"
            str_id += 1
        slots[ph] = (value, quote_mark)
        # update intent to use the placeholder
        quoted_value = quote_mark + value + quote_mark
        intent = intent.replace(quoted_value, ph)
    return intent, slots


def process_snippet(snippet, slots):
    """Replace strings, identifiers, and iterables in snippet with placeholders."""
    # replace strings and identifiers by walking the AST
    pyast = ast.parse(snippet)
    for node in ast.walk(pyast):
        for k, v in vars(node).items():
            if k in ("lineno", "col_offset", "ctx"):
                continue
            for ph, (value, _) in slots.items():
                if v == value:
                    setattr(node, k, ph)
    snippet = ast.unparse(pyast)

    # replace iterables by string search
    for ph, (value, _) in slots.items():
        if len(value) > 0 and value[0] in ("[", "(", "{"):
            snippet = snippet.replace(value, ph)

    return snippet


def unprocess_snippet(snippet, slots):
    """Revert a processed snippet to original."""
    # revert strings and identifiers by walking the AST
    pyast = ast.parse(snippet)
    for node in ast.walk(pyast):
        for k, v in vars(node).items():
            if k in ("lineno", "col_offset", "ctx"):
                continue
            for ph, (value, _) in slots.items():
                if v == ph:
                    setattr(node, k, value)
    snippet = ast.unparse(pyast)

    # revert everything else by string search
    for ph, (value, _) in slots.items():
        snippet = snippet.replace(ph, value)

    return snippet


def process_and_filter(intent_snippets):
    """Process all intents and snippet, then filter the result by testing roundtrip.

    Return a list of (processed_intent, processed_snippet, slots) triples."""
    triples = []
    for intent, snippet in intent_snippets:
        # parse and unparse snippet to standarize format
        snippet = ast.unparse(ast.parse(snippet))
        intent1, slots = process_intent(intent)
        snippet1 = process_snippet(snippet, slots)
        try:
            snippet2 = unprocess_snippet(snippet1, slots)
        except SyntaxError:
            continue
        if snippet == snippet2:
            triples.append((intent1, snippet1, slots))
    return triples


def tokenize_intent(intent):
    """Tokenize intent with nltk tokenizer."""
    return word_tokenize(intent.lower())


class ConalaDataset:
    def __init__(
        self,
        filepath,
        grammar,
        max_sentence_len,
        max_recipe_len,
        intent_freq_cutoff,
        action_freq_cutoff,
        intent_vocab=None,
        action_vocab=None,
    ):
        self.intent_snippets = load_conala_from_json(filepath)
        self.intent_snippet_slots = process_and_filter(self.intent_snippets)

        if intent_vocab:
            self.intent_vocab = intent_vocab
        else:
            corpus = [
                token
                for intent, _, _ in self.intent_snippet_slots
                for token in tokenize_intent(intent)
            ]
            self.intent_vocab = Vocab(
                corpus, freq_cutoff=intent_freq_cutoff, special_words=["<sos>", "<eos>"]
            )
            print(f"Intent Vocab: {len(self.intent_vocab)}/{len(set(corpus))}")

        if action_vocab:
            self.action_vocab = action_vocab
        else:
            mrs = [
                ast_to_mr(ast.parse(snippet))
                for _, snippet, _ in self.intent_snippet_slots
            ]
            action_corpus = [
                action for mr in mrs for action in mr_to_recipe_dfs(mr, grammar)
            ]
            self.action_vocab = Vocab(
                action_corpus,
                freq_cutoff=action_freq_cutoff,
                special_words=["<soa>", "<eoa>"],
            )
            print(f"Action Vocab: {len(self.action_vocab)}/{len(set(action_corpus))}")

        self.sentences = []
        self.recipes = []
        for intent, snippet, _ in self.intent_snippet_slots:
            tokens = tokenize_intent(intent)
            tokens = ["<sos>"] + tokens + ["<eos>"]
            token_ids = torch.tensor(
                [self.intent_vocab.word2id(token) for token in tokens]
            )
            actions = mr_to_recipe_dfs(ast_to_mr(ast.parse(snippet)), grammar)
            actions = ["<soa>"] + actions + ["<eoa>"]
            action_ids = torch.tensor(
                [self.action_vocab.word2id(action) for action in actions]
            )
            if len(tokens) <= max_sentence_len and len(actions) <= max_recipe_len:
                self.sentences.append(token_ids)
                self.recipes.append(action_ids)

    def __getitem__(self, idx):
        return self.sentences[idx], self.recipes[idx]

    def __len__(self):
        return len(self.sentences)

    def add_argparse_args(parser):
        group = parser.add_argument_group("ConalaDataset")
        group.add_argument("--max_sentence_len", type=int, default=30)
        group.add_argument("--max_recipe_len", type=int, default=80)
        group.add_argument("--intent_freq_cutoff", type=int, default=2)
        group.add_argument("--action_freq_cutoff", type=int, default=2)
        return parser
