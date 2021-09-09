#!/usr/env/bin python3
import re

from conala import load_intent_snippet

# extract all quote-snippet pairs
quotes = []
snippets = []
for intent, snippet in load_intent_snippet('data/conala-train.json'):
    patterns = [
        r"`.*?`",
        r"'.*?'",
        r'".*?"',
    ]
    for pat in patterns:
        for quote in re.findall(pat, intent):
            # strip the actual quotation marks
            quote = quote[1:-1]
            # strip whitespace
            quote = quote.strip()
            quotes.append(quote)
            snippets.append(snippet)


def is_empty(snippet, quote):
    return len(quote) == 0


def is_string_literal(snippet, quote):
    return len(quote) > 0 and f"'{quote}'" in snippet or f'"{quote}"' in snippet


def is_identifier(snippet, quote):
    return (
        quote.isidentifier()
        and quote in snippet
        and not is_string_literal(snippet, quote)
    )


def is_list(snippet, quote):
    return len(quote) > 0 and quote[0] == "["


def is_dict_or_set(snippet, quote):
    return len(quote) > 0 and quote[0] == "{"


def classify(cls_fns, snippets, quotes):
    count = {}
    items = {}
    for cls, fn in cls_fns.items():
        count[cls] = sum(
            1 for snippet, quote in zip(snippets, quotes) if fn(snippet, quote)
        )
        items[cls] = [
            (snippet, quote)
            for snippet, quote in zip(snippets, quotes)
            if fn(snippet, quote)
        ]
    count["accounted"] = sum(count.values())
    count["total"] = len(snippets)
    items["unaccounted"] = [
        (snippet, quote)
        for snippet, quote in zip(snippets, quotes)
        if not any(fn(snippet, quote) for fn in cls_fns.values())
    ]
    count["unaccounted"] = len(items["unaccounted"])
    items["intersect"] = [
        (snippet, quote)
        for snippet, quote in zip(snippets, quotes)
        for fn1 in cls_fns.values()
        for fn2 in cls_fns.values()
        if fn1 != fn2
        if fn1(snippet, quote) and fn2(snippet, quote)
    ]
    return count, items


count, items = classify(
    {
        "empty": is_empty,
        "identifier": is_identifier,
        "string-literal": is_string_literal,
        "list": is_list,
        "dict/set": is_dict_or_set,
    },
    snippets,
    quotes,
)