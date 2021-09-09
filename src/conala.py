# load and preprocess CoNaLa dataset

import json
import re
import ast

from asdl.convert import ast_to_mr
from asdl.replace import replace_mr
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
    #   3. generate target MR in the following ways(read on for explanation):
    #        parse quoted content as a string
    #        parse quoted content as an expression
    #        parse quoted content as an expression, but change 'ctx': 'Load' to 'ctx': 'Store'
    #   4. replace target MR to {"_tag": "placeholder", "value": "<ph_0>"}
    #
    # Why parse quoted content as string first?
    # Usually, code is quoted with backticks and strings are quoted with single/double quotes.
    # However, sometimes single/double quotes are used with code as well, and vice versa.
    # So we'll always try matching string literals, then fallback to code.
    #
    # Why change ctx to 'Store'?
    # A variable can be used in two context: 'Load' or 'Store'.
    # But parsing a variable only generates 'Load'; we need to handle 'Store' as well.

    ph2mr = {} # map placeholders to original MR
    quote2ph = {} # map quoted content to placeholder(to resolve duplicates)

    def extract(parsed_mr):
        """Extract inner mr from parsed mr."""
        # assume inner code is an expression
        try:
            if parsed_mr["body"][0]["_tag"] != "Expr":
                raise SyntaxError("Quoted content is not an expression.")
        except (IndexError, KeyError):
            raise SyntaxError("Quoted content is not an expression.")
        return parsed_mr["body"][0]["value"]

    def generate_placeholder(match):
        nonlocal mr
        quote = match.group()
        if quote in quote2ph:
            return quote2ph[quote]
        else:
            ph = f"<ph_{len(ph2mr)}>"
            quote2ph[quote] = ph
        tagged_ph = {"_tag": "placeholder", "value": ph}
        # use lambda to delay evaluation
        target_fns = [
            lambda: extract(ast_to_mr(ast.parse(f"'{quote[1:-1]}'"))),  # parse as string
            lambda: extract(ast_to_mr(ast.parse(quote[1:-1]))),  # parse as expression
            lambda: dict(
                extract(ast_to_mr(ast.parse(quote[1:-1]))), ctx=dict(_tag="Store")
            ),  # parse as variable with 'ctx': 'Store'
        ]
        for target_fn in target_fns:
            target = target_fn()
            mr, found = replace_mr(mr, target, tagged_ph)
            if found:
                ph2mr[ph] = target
                return ph
        raise SyntaxError("Cannot match var/literal between intent and snippet")

    # deal with backticks first because backticks may contain single/double quotes
    intent = re.sub(r"`.*?`", generate_placeholder, intent)
    intent = re.sub(r'".*?"', generate_placeholder, intent)
    intent = re.sub(r"'.*?'", generate_placeholder, intent)
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
