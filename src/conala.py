# load and preprocess CoNaLa dataset

import json
import re
import ast

from asdl.convert import ast_to_mr
from asdl.transform import transform_mr


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

    Returns (new_intent, new_mr, ph2mr) where ph2mr is a dictionary that maps placeholders
    to original MR representation in the snippet.

    Note that new_mr may share parts with the original mr. Handle with care.

    Raises SyntaxError if anything goes wrong.
    """
    # For each pair of quotes(single/double/backtick) in the intent, do the followings:
    #   1. generate a placeholder, such as <ph_0>
    #   2. replace quoted content from the intent with the placeholder
    #   3. parse quoted content to MR. include single/double quotes but not backticks when parsing
    #   4. replace target MR to {"_tag": "placeholder", "value": "<ph_0>"}
    ph2mr = {}

    def generate_placeholder(match):
        placeholder = f"<ph_{len(ph2mr)}>"
        quoted_content = match.group(1)
        mr = ast_to_mr(ast.parse(quoted_content))
        # usually quoted_content is an expression
        if mr["body"][0]["_tag"] != "Expr":
            raise SyntaxError("Quoted content is not an expression.")
        ph2mr[placeholder] = mr["body"][0]["value"]
        return placeholder

    # deal with backticks first because backticks may contain single/double quotes
    intent = re.sub(r"`(.*)`", generate_placeholder, intent)
    intent = re.sub(r'(".*")', generate_placeholder, intent)
    intent = re.sub(r"('.*')", generate_placeholder, intent)

    for placeholder, target_mr in ph2mr.items():
        tagged_placeholder = {"_tag": "placeholder", "value": placeholder}
        mr = transform_mr(mr, lambda mr: tagged_placeholder if mr == target_mr else mr)

    return intent, mr, ph2mr


def uncanonicalize(mr, ph2mr):
    """Replace placeholders in MR back to original.

    Returns new_mr that can be converted back to a valid AST.

    Note that new_mr may share parts with the original mr. Handle with care.
    """
    for placeholder, target_mr in ph2mr.items():
        tagged_placeholder = {"_tag": "placeholder", "value": placeholder}
        mr = transform_mr(mr, lambda mr: target_mr if mr == tagged_placeholder else mr)

    return mr
