import pytest
import ast

from asdl.convert import ast_to_mr
from conala import canonicalize, uncanonicalize


def test_canonicalize():
    intent = "Hello `variable`, 'string'."
    snippet = "print(variable, file='string')"
    mr = ast_to_mr(ast.parse(snippet))
    new_intent, new_mr, ph2mr = canonicalize(intent, mr)
    print(new_mr)
    assert new_intent == "Hello <ph_0>, <ph_1>."
    assert new_mr['body'][0]['value']['args'][0]['_tag'] == 'placeholder'
    assert uncanonicalize(new_mr, ph2mr) == mr
    assert len(ph2mr) == 2