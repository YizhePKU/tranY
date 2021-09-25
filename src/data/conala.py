"""Load the CoNaLa dataset."""

import ast
import json
import re
import random
import nltk
from copy import deepcopy

import cfg
import torch
import torch.nn.functional as F
from asdl.convert import ast_to_mr
from asdl.recipe import mr_to_recipe_dfs
from asdl.utils import tagged, walk
from utils.flatten import flatten
from data.tokenizer import Vocab
from data.preprocess import preprocess_example


class ConalaDataset(torch.utils.data.Dataset):
    """Load the CoNaLa dataset.

    The entire dataset is loaded into memory since it's tiny(~500KB).

    Outputs:
        words (*): ids of input words, including SOS and EOS.
        label (cfg.max_recipe_len): ids of output recipes, including SOA and EOA.
    """

    def __init__(
        self, filepath, grammar, rewrite_intent="when-available", special_tokens=[],
        intent_vocab=None, action_vocab=None, shuffle=True
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

        # Preprocess: replace strings and variables to `stri`, `vari`
        processed_data = [preprocess_example(self.intents[i], self.snippets[i]) for i in range(len(self.intents))]
        orig_len = len(processed_data)
        # filter invalid canonical_snippet:
        def filter_func(dic):
            try:
                ast.parse(dic["canonical_snippet"])
                return True
            except:
                return False
        processed_data = list(filter(filter_func, processed_data))
        if shuffle:
            processed_data = random.sample(processed_data, len(processed_data))
        new_len = len(processed_data)
        print(f"Passed {orig_len - new_len} invalid code.")
        self.c_intents = [dic["canonical_intent"] for dic in processed_data]            # not tokenized, str
        self.intent_tokens = [dic["intent_tokens"] for dic in processed_data]           # tokenized, list
        self.slot_map = [dic["slot_map"] for dic in processed_data]
        self.c_code = [dic["canonical_snippet"] for dic in processed_data]
        # convert snippet to MR
        self.c_mrs = [ast_to_mr(ast.parse(snippet)) for snippet in self.c_code]

        # convert MR to recipes
        self.recipes = [mr_to_recipe_dfs(mr, grammar) for mr in self.c_mrs]

        # build intent vocab
        if intent_vocab:
            # use vocab from training set
            self.intent_vocab = intent_vocab
        else:
            self.intent_vocab = Vocab.from_corpus(special_tokens, self.intent_tokens, size=99999)
        self.intent2id = self.intent_vocab.word2id
        self.id2intent = self.intent_vocab.id2word

        # build action vocab
        if action_vocab:
            # use vocab from training set
            self.action_vocab = action_vocab
        else:
            self.action_vocab = Vocab.from_corpus(special_tokens, self.recipes, size=99999)
        self.action2id = self.action_vocab.word2id
        self.id2action = self.action_vocab.id2word
        self.action_vocab_size = len(self.id2action)

        self.sentences = []
        self.labels = []
        for index in range(len(self.c_code)):
            c_intent = self.c_intents[index]
            sentence = torch.tensor(self.convert_intent_ids(c_intent), dtype=torch.long)
            self.sentences.append(sentence)
            recipe = ["[SOA]"] + self.recipes[index] + ["[EOA]"]
            label = torch.tensor(
                [self.action2id[action] for action in recipe], dtype=torch.long
            )
            label = F.pad(label, (self.action_vocab.pad_id, cfg.max_recipe_len - len(recipe)))
            self.labels.append(label)

    def convert_intent_ids(self, intent):
        """
        Args: intent is a word list
        """
        if isinstance(intent, str):
            intent = nltk.word_tokenize(intent)
        return [self.intent2id[word] if word in self.intent2id else self.intent_vocab.unk_id for word in intent]
    
    def convert_ids_intent(self, ids):
        """
        Args: ids is an id list
        """
        return [self.id2intent[id] if id < len(self.id2intent) else "[UNK]" for id in ids]

    def convert_action_ids(self, action):
        return [self.action2id[word] if word in self.action2id else self.action_vocab.unk_id for word in action]
    
    def convert_ids_action(self, ids):
        return [self.id2action[id] if id < len(self.id2action) else "[UNK]" for id in ids]
    

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.intents)