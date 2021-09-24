"""Load the CoNaLa dataset."""

import ast
import json
import re
from copy import deepcopy

import cfg
import torch
import torch.nn.functional as F
from asdl.convert import ast_to_mr
from asdl.recipe import mr_to_recipe_dfs
from asdl.utils import tagged, walk
from utils.flatten import flatten

from data.tokenizer import make_lookup_tables, train_intent_tokenizer


class ConalaDataset(torch.utils.data.Dataset):
    """Load the CoNaLa dataset.

    The entire dataset is loaded into memory since it's tiny(~500KB).

    Outputs:
        words (*): ids of input words, including SOS and EOS.
        label (cfg.max_recipe_len): ids of output recipes, including SOA and EOA.
    """

    def __init__(
        self, filepath, grammar, rewrite_intent="when-available", special_tokens=[]
    ):
        """
        Args:
            filepath: JSON filepath of CoNaLa data.
            grammar (asdl.parser.Module): Python ASDL grammar definition.
            rewrite_intent: whether to use rewritten_intent when available.
                Allowed values are 'never', 'when-available', 'skip-when-unavailable'
            special_tokens (list[str]): special tokens to insert into vocabularies.
        """
        with open(filepath) as file:
            data = json.load(file)

        # extract intents and snippets
        self.intents = []
        self.snippets = []
        for sample in data:
            intent = sample["intent"]
            snippet = sample["snippet"]
            if (
                sample["rewritten_intent"] is None
                and rewrite_intent == "skip-when-unavailable"
            ):
                continue
            if (
                sample["rewritten_intent"] is not None
                and rewrite_intent == "when-available"
            ):
                intent = sample["rewritten_intent"]
            self.intents.append(intent)
            self.snippets.append(snippet)

        # convert snippet to MR
        self.mrs = [ast_to_mr(ast.parse(snippet)) for snippet in self.snippets]

        # canonicalize intent and MR
        # prefix "c" stands for canonicalized
        self.c_intents, self.c_mrs, self.ph2mrs = zip(
            *(canonicalize(intent, mr) for intent, mr in zip(self.intents, self.mrs))
        )

        # convert MR to recipes
        self.recipes = [list(mr_to_recipe_dfs(mr, grammar)) for mr in self.c_mrs]

        # train an intent tokenizer
        self.intent_tokenizer = train_intent_tokenizer(
            self.c_intents, special_tokens=special_tokens
        )

        # make action loopup tables
        self.id2action, self.action2id = make_lookup_tables(
            flatten(self.recipes),
            special_tokens=special_tokens,
        )

        self.word_vocab_size = self.intent_tokenizer.get_vocab_size()
        self.action_vocab_size = len(self.id2action)

    def __getitem__(self, index):
        c_intent = self.c_intents[index]
        sentence = torch.tensor(self.intent_tokenizer.encode(c_intent).ids)

        recipe = ["[SOA]"] + self.recipes[index] + ["[EOA]"]
        label = torch.tensor(
            [self.action2id[action] for action in recipe], dtype=torch.long
        )
        label = F.pad(label, (0, cfg.max_recipe_len - len(recipe)))
        return sentence, label

    def __len__(self):
        return len(self.intents)


def canonicalize(intent, mr):
    """Replace free variables and literals in intent and MR with placeholders.

    This is necessary because the intent often includes variables and string literals that
    must be copied into the generated snippet. Replacing them with placeholders makes it
    more likely for the tranX model to invoke the copy mechanism.

    Returns (new_intent, new_mr, ph2mr) where ph2mr is a dictionary that maps placeholders
    to original MR representation in the snippet.
    """
    # For each pair of quotes(single/double/backtick) in the intent, do the followings:
    #   1. generate a placeholder, such as <ph_0>
    #   2. replace quoted content from the intent with the placeholder
    #   3. walk target MR and update various tags in place:
    #        identifiers -- replace with `placeholder`
    #        string literals -- replace with "placeholder"
    #        lists, dicts, and sets -- no-op for now
    #
    # This replacement strategy covers ~90% of quotes in the training set.

    new_mr = deepcopy(mr)  # make a copy so that we can modify MR in place
    ph2mr = {}  # map placeholders to original parts of the MR
    quote2ph = {}  # map quotes to placeholders to handle duplicated quotes

    def generate_placeholder(match):
        quote = match.group()[1:-1]
        # returns immediately if we've processed this quote already
        if quote in quote2ph:
            return quote2ph[quote]
        # otherwise generate a new placeholder and proceed
        ph = f"<ph_{len(ph2mr)}>"
        quote2ph[quote] = ph
        for node in walk(new_mr):
            # replace identifiers
            if tagged(node, "Name") and node["id"] == quote:
                ph2mr[ph] = node["id"]
                node["id"] = ph
            # replace string literals
            if tagged(node, "Constant") and node["value"] == quote:
                ph2mr[ph] = node["value"]
                node["value"] = ph
        return ph

    # FIXME: regular doesn't work when intent contains escaped quotes
    intent = re.sub(r"`.*?`", generate_placeholder, intent)
    intent = re.sub(r'".*?"', generate_placeholder, intent)
    intent = re.sub(r"'.*?'", generate_placeholder, intent)
    return intent, new_mr, ph2mr


def uncanonicalize(mr, ph2mr):
    """Replace placeholders in MR back to the original.

    Returns a MR that can be converted back to a valid AST.
    """
    new_mr = deepcopy(mr)  # make a copy so that we can modify MR in place
    for placeholder, target in ph2mr.items():
        for node in walk(new_mr):
            # replace identifiers
            if tagged(node, "Name") and node["id"] == placeholder:
                node["id"] = target
            # replace string literals
            if tagged(node, "Constant") and node["value"] == placeholder:
                node["value"] = target
    return new_mr
