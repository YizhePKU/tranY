# load and preprocess CoNaLa dataset

import json
import re

from asdl.transform import transform_mr
from asdl.utils import walk, tagged


def load_intent_snippet(filepath, rewrite_intent="when-available"):
    """Load (rewritten_intent, snippet) pairs.

    Parameters:
        filepath: path to JSON data
        rewrite_intent: whether to use rewritten_intent when available.
            Allowed values are 'never', 'when-available', 'skip-when-unavailable'
    """
    with open(filepath) as file:
        data = json.load(file)
    pairs = []
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
        pairs.append((intent, snippet))
    return pairs


def canonicalize(intent, mr):
    """Replace free variables and literals in intent and MR with placeholders.

    This is necessary because the intent often includes variables and string literals that
    must be copied into the generated snippet. Replacing them with placeholders makes it
    more likely for the tranX model to invoke the copy mechanism.

    Returns (new_intent, mr, ph2mr) where ph2mr is a dictionary that maps placeholders
    to original MR representation in the snippet.

    Note that mr is updated in place.

    Raises SyntaxError if anything goes wrong.
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

    ph2mr = {}  # map placeholders to original MR
    quote2ph = {}  # map quoted content to placeholder(to resolve duplicates)

    def generate_placeholder(match):
        quote = match.group()[1:-1]
        # returns immediately if we've processed this quote already
        if quote in quote2ph:
            return quote2ph[quote]
        # otherwise generate a new placeholder and proceed
        ph = f"<ph_{len(ph2mr)}>"
        quote2ph[quote] = ph
        for node in walk(mr):
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
    return intent, mr, ph2mr


def uncanonicalize(mr, ph2mr):
    """Replace placeholders in MR back to the original.

    Returns mr that can be converted back to a valid AST.

    Note that mr is updated in place.
    """
    for placeholder, target in ph2mr.items():
        for node in walk(mr):
            # replace identifiers
            if tagged(node, "Name") and node["id"] == placeholder:
                node["id"] = target
            # replace string literals
            if tagged(node, "Constant") and node["value"] == placeholder:
                node["value"] = target
    return mr
