import ast

import pytest
from asdl.convert import ast_to_mr
from asdl.parser import parse as parse_asdl

from data.conala import ConalaDataset, canonicalize, uncanonicalize


@pytest.fixture
def grammar():
    return parse_asdl("src/asdl/Python.asdl")


def test_canonicalize():
    intent = "Hello `variable`, 'string'."
    snippet = "print(variable, file='string')"
    mr = ast_to_mr(ast.parse(snippet))
    new_intent, new_mr, ph2mr = canonicalize(intent, mr)
    assert new_intent == "Hello <ph_0>, <ph_1>."
    assert new_mr["body"][0]["value"]["args"][0]["id"] == "<ph_0>"
    assert uncanonicalize(new_mr, ph2mr) == mr
    assert len(ph2mr) == 2
    assert mr is not new_mr  # make sure a copy takes place


@pytest.mark.skip(reason="lists/set/dict not supported yet")
def test_canonicalize_double_backticks():
    intent = "`[1, 2]` and `[3, 4]`"
    snippet = "print([1, 2], [3, 4])"
    mr = ast_to_mr(ast.parse(snippet))
    new_intent, new_mr, ph2mr = canonicalize(intent, mr)
    assert new_intent == "<ph_0> and <ph_1>"
    assert uncanonicalize(new_mr, ph2mr) == mr
    assert len(ph2mr) == 2


def test_canonicalize_variable_store():
    intent = "`var1` and 'var2'"
    snippet = "var1 = var2"
    mr = ast_to_mr(ast.parse(snippet))
    new_intent, new_mr, ph2mr = canonicalize(intent, mr)
    assert new_intent == "<ph_0> and <ph_1>"
    assert uncanonicalize(new_mr, ph2mr) == mr
    assert len(ph2mr) == 2


def test_canonicalize_weird_string():
    intent = "Hello 'ef-sv-3=:'"
    snippet = "print('ef-sv-3=:')"
    mr = ast_to_mr(ast.parse(snippet))
    new_intent, new_mr, ph2mr = canonicalize(intent, mr)
    assert new_intent == "Hello <ph_0>"
    assert uncanonicalize(new_mr, ph2mr) == mr
    assert len(ph2mr) == 1


def test_canonicalize_duplicate_intent():
    intent = "create `result` from `list1` if first element of `list1` is in `list2`"
    snippet = "result = [x for x in list1 if x[0] in list2]"
    mr = ast_to_mr(ast.parse(snippet))
    new_intent, new_mr, ph2mr = canonicalize(intent, mr)
    assert (
        new_intent
        == "create <ph_0> from <ph_1> if first element of <ph_1> is in <ph_2>"
    )
    assert uncanonicalize(new_mr, ph2mr) == mr
    assert len(ph2mr) == 3


def test_canonicalize_roundtrip(grammar):
    ds = ConalaDataset("data/conala-train.json", grammar=grammar)
    for intent, snippet in zip(ds.intents, ds.snippets):
        pyast = ast.parse(snippet)
        mr = ast_to_mr(pyast)
        new_intent, new_mr, ph2mr = canonicalize(intent, mr)
        restored_mr = uncanonicalize(new_mr, ph2mr)
        assert new_mr == mr
