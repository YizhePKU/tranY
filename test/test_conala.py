import pytest
import ast

from asdl.convert import ast_to_mr
from conala import canonicalize, uncanonicalize


def test_canonicalize():
    intent = "Hello `variable`, 'string'."
    snippet = "print(variable, file='string')"
    mr = ast_to_mr(ast.parse(snippet))
    new_intent, new_mr, ph2mr = canonicalize(intent, mr)
    assert new_intent == "Hello <ph_0>, <ph_1>."
    assert new_mr["body"][0]["value"]["args"][0]["_tag"] == "placeholder"
    assert uncanonicalize(new_mr, ph2mr) == mr
    assert len(ph2mr) == 2


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