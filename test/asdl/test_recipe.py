import ast

import pytest
from data.conala_v2 import ConalaDataset

from asdl.convert import ast_to_mr
from asdl.parser import parse as parse_asdl
from asdl.recipe import (
    Builder,
    mr_to_recipe_dfs,
    preprocess_grammar,
    recipe_to_mr_dfs,
)


@pytest.fixture
def grammar():
    return parse_asdl("src/asdl/python3_simplified.asdl")


def test_preprocess_grammar(grammar):
    type2constr, constr2type, fields, name2fields = preprocess_grammar(grammar)
    assert type2constr["mod"] == ["Module", "Interactive", "Expression", "FunctionType"]
    assert type2constr["arguments"] == ["arguments"]
    assert constr2type["Raise"] == "stmt"
    assert constr2type["keyword"] == "keyword"
    assert fields["Assert"] == [
        ("expr", "test", "single"),
        ("expr", "msg", "optional"),
    ]
    assert fields["comprehension"] == [
        ("expr", "target", "single"),
        ("expr", "iter", "single"),
        ("expr", "ifs", "multiple"),
        ("int", "is_async", "single"),
    ]
    assert name2fields["FunctionDef"]["args"] == ("arguments", "args", "single")
    assert name2fields["ExceptHandler"]["body"] == ("stmt", "body", "multiple")


def test_mr_to_recipe_dfs_variable(grammar):
    mr = {
        "_tag": "Name",
        "id": "x",
    }
    recipe = mr_to_recipe_dfs(mr, grammar)
    assert recipe == [
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
    ]


def test_mr_to_recipe_dfs_assignment(grammar):
    mr = {
        "_tag": "Assign",
        "targets": [
            {
                "_tag": "Name",
                "id": "x",
            }
        ],
        "value": {
            "_tag": "Constant",
            "value": 1,
            "kind": None,
        },
        "type_comment": None,
    }
    recipe = mr_to_recipe_dfs(mr, grammar)
    assert recipe == [
        ("ApplyConstr", "Assign"),
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", 1),
        ("Reduce",),
        ("Reduce",),
    ]


def test_mr_to_recipe_dfs(grammar):
    snippet = "panda.read_csv('file.csv', nrows=100)"
    mr = ast_to_mr(ast.parse(snippet))
    mr = mr["body"][0]  # strip the outer 'module' tag
    recipe = mr_to_recipe_dfs(mr, grammar)
    assert recipe == [
        ("ApplyConstr", "Expr"),
        ("ApplyConstr", "Call"),
        ("ApplyConstr", "Attribute"),
        ("ApplyConstr", "Name"),
        ("GenToken", "panda"),
        ("GenToken", "read_csv"),
        ("ApplyConstr", "Constant"),  # "Str" instead of "Constant" in paper
        ("GenToken", "file.csv"),
        # ("GenToken", "</f>"), # in paper
        ("Reduce",),  # Constant.kind, not in paper
        ("Reduce",),
        ("ApplyConstr", "keyword"),
        ("GenToken", "nrows"),
        ("ApplyConstr", "Constant"),  # "Num" instead of "Constant" in paper
        ("GenToken", 100),
        ("Reduce",),  # Constant.kind, not in paper
        ("Reduce",),
    ]


def test_recipe_to_mr_dfs_variable(grammar):
    recipe = [
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
    ]
    mr = recipe_to_mr_dfs(recipe, grammar)
    assert mr == {
        "_tag": "Name",
        "id": "x",
    }


def test_recipe_to_mr_dfs_assignment(grammar):
    recipe = [
        ("ApplyConstr", "Assign"),
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", 1),
        ("Reduce",),
        ("Reduce",),
    ]
    mr = recipe_to_mr_dfs(recipe, grammar)
    assert mr == {
        "_tag": "Assign",
        "targets": [
            {
                "_tag": "Name",
                "id": "x",
            }
        ],
        "value": {
            "_tag": "Constant",
            "value": 1,
            "kind": None,
        },
        "type_comment": None,
    }


def test_mr_to_recipe_dfs_list(grammar):
    mr = {
        "_tag": "List",
        "elts": [
            {"_tag": "Constant", "value": 1, "kind": None},
            {"_tag": "Constant", "value": 2, "kind": None},
        ],
    }
    recipe = mr_to_recipe_dfs(mr, grammar)
    assert recipe == [
        ("ApplyConstr", "List"),
        ("ApplyConstr", "Constant"),
        ("GenToken", 1),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", 2),
        ("Reduce",),
        ("Reduce",),
    ]


def test_recipe_to_mr_dfs_list(grammar):
    recipe = [
        ("ApplyConstr", "List"),
        ("ApplyConstr", "Constant"),
        ("GenToken", 1),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", 2),
        ("Reduce",),
        ("Reduce",),
    ]
    mr = recipe_to_mr_dfs(recipe, grammar)
    assert mr == {
        "_tag": "List",
        "elts": [
            {"_tag": "Constant", "value": 1, "kind": None},
            {"_tag": "Constant", "value": 2, "kind": None},
        ],
    }


def test_recipe_dfs_roundtrip(grammar):
    ds = ConalaDataset(
        "data/conala-train.json",
        grammar,
        max_sentence_len=40,
        max_recipe_len=100,
        intent_freq_cutoff=2,
        action_freq_cutoff=2,
    )
    for intent, snippet in ds.intent_snippets:
        pyast = ast.parse(snippet)
        mr = ast_to_mr(pyast)
        recipe = mr_to_recipe_dfs(mr, grammar)
        reconstructed_mr = recipe_to_mr_dfs(recipe, grammar)
        assert mr == reconstructed_mr


def test_builder_simple(grammar):
    builder = Builder(grammar)
    recipe = [
        ("ApplyConstr", "Expression"),
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
    ]

    builder0 = builder.apply_action(recipe[0])
    assert builder0.result == {"_tag": "Expression"}

    builder1 = builder0.apply_action(recipe[1])
    assert builder1.result == {"_tag": "Expression", "body": {"_tag": "Name"}}

    builder2 = builder1.apply_action(recipe[2])
    assert builder2.result == {
        "_tag": "Expression",
        "body": {"_tag": "Name", "id": "x"},
    }


def test_builder_assignment(grammar):
    recipe = [
        ("ApplyConstr", "Module"),
        ("ApplyConstr", "Assign"),
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", 1),
        ("Reduce",),  # close Constant.kind
        ("Reduce",),  # close assign.msg
        ("Reduce",),  # close Module.body
        ("Reduce",),  # close Module.type_ignores
    ]
    builder = Builder(grammar)

    for i, action in enumerate(recipe):
        print(i)
        builder = builder.apply_action(action)

    assert builder.done
    assert builder.result == {
        "_tag": "Module",
        "body": [
            {
                "_tag": "Assign",
                "targets": [
                    {
                        "_tag": "Name",
                        "id": "x",
                    }
                ],
                "value": {
                    "_tag": "Constant",
                    "value": 1,
                    "kind": None,
                },
                "type_comment": None,
            }
        ],
        "type_ignores": [],
    }


def test_builder_empty(grammar):
    builder = Builder(grammar)
    assert builder.done == False
    assert builder.result is None


def test_builder_wrong_type(grammar):
    builder = Builder(grammar)
    with pytest.raises(ValueError):
        builder.apply_action(("GenToken", "Hello"))


def test_builder_too_many_actions(grammar):
    recipe = [
        ("ApplyConstr", "Interactive"),
        ("Reduce",),
    ]
    builder = Builder(grammar)
    for action in recipe:
        builder = builder.apply_action(action)
    with pytest.raises(ValueError):
        builder.apply_action(("GenToken", "World"))


def test_builder_allowed_actions(grammar):
    builder = Builder(grammar)
    builder = builder.apply_action(("ApplyConstr", "Expression"))
    assert ("ApplyConstr", "IfExp") in builder.allowed_actions
    assert len(builder.allowed_actions) == 27

    builder = builder.apply_action(("ApplyConstr", "Constant"))
    assert ("GenToken",) in builder.allowed_actions
    assert len(builder.allowed_actions) == 1


def test_builder_roundtrip(grammar):
    ds = ConalaDataset(
        "data/conala-train.json",
        grammar,
        max_sentence_len=40,
        max_recipe_len=100,
        intent_freq_cutoff=2,
        action_freq_cutoff=2,
    )
    for intent, snippet in ds.intent_snippets:
        pyast = ast.parse(snippet)
        mr = ast_to_mr(pyast)
        recipe = mr_to_recipe_dfs(mr, grammar)
        builder = Builder(grammar)
        for action in recipe:
            builder = builder.apply_action(action)
        reconstructed_mr = builder.result
        assert mr == reconstructed_mr
