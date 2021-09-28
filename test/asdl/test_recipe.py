import ast

import pytest
from data.conala import ConalaDataset

from asdl.convert import ast_to_mr
from asdl.parser import parse as parse_asdl
from asdl.recipe import (Builder, extract_cardinality, int2str,
                         mr_to_recipe_dfs, preprocess_grammar,
                         recipe_to_mr_dfs, str2int)


@pytest.fixture
def grammar():
    return parse_asdl("src/asdl/Python.asdl")


def test_cardinality(grammar):
    cardinality = extract_cardinality(grammar)
    assert cardinality["Module"]["body"] == "multiple"
    assert cardinality["ClassDef"]["name"] == "single"
    assert cardinality["keyword"]["arg"] == "optional"
    assert cardinality["arguments"]["defaults"] == "multiple"


def test_cardinality_field_order(grammar):
    cardinality = extract_cardinality(grammar)
    assert list(cardinality["Module"].keys()) == ["body", "type_ignores"]
    assert list(cardinality["AsyncFor"].keys()) == [
        "target",
        "iter",
        "body",
        "orelse",
        "type_comment",
    ]


def test_preprocess_grammar(grammar):
    type2constr, constr2type, fields = preprocess_grammar(grammar)
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


def test_mr_to_recipe_dfs_variable(grammar):
    mr = {
        "_tag": "Name",
        "id": "x",
        "ctx": {"_tag": "Store"},
    }
    recipe = mr_to_recipe_dfs(mr, grammar)
    assert recipe == [
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("ApplyConstr", "Store"),
    ]


def test_mr_to_recipe_dfs_assignment(grammar):
    mr = {
        "_tag": "Assign",
        "targets": [
            {
                "_tag": "Name",
                "id": "x",
                "ctx": {"_tag": "Store"},
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
        ("ApplyConstr", "Store"),
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
        ("ApplyConstr", "Load"),  # Name.ctx, not in paper
        ("GenToken", "read_csv"),
        ("ApplyConstr", "Load"),  # Name.ctx, not in paper
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
        ("ApplyConstr", "Store"),
    ]
    mr = recipe_to_mr_dfs(recipe, grammar)
    assert mr == {
        "_tag": "Name",
        "id": "x",
        "ctx": {"_tag": "Store"},
    }


def test_recipe_to_mr_dfs_assignment(grammar):
    recipe = [
        ("ApplyConstr", "Assign"),
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("ApplyConstr", "Store"),
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
                "ctx": {"_tag": "Store"},
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
        "ctx": {"_tag": "Load"},
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
        ("ApplyConstr", "Load"),
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
        ("ApplyConstr", "Load"),
    ]
    mr = recipe_to_mr_dfs(recipe, grammar)
    assert mr == {
        "_tag": "List",
        "elts": [
            {"_tag": "Constant", "value": 1, "kind": None},
            {"_tag": "Constant", "value": 2, "kind": None},
        ],
        "ctx": {"_tag": "Load"},
    }


def test_recipe_dfs_roundtrip(grammar):
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SOA]", "[EOA]"]
    ds = ConalaDataset(
        "data/conala-train.json", grammar=grammar, special_tokens=special_tokens
    )
    for intent, snippet in zip(ds.intents, ds.snippets):
        pyast = ast.parse(snippet)
        mr = ast_to_mr(pyast)
        recipe = mr_to_recipe_dfs(mr, grammar)
        reconstructed_mr = recipe_to_mr_dfs(recipe, grammar)
        assert mr == reconstructed_mr


def test_int2str():
    recipe = [
        ("ApplyConstr", "List"),
        ("ApplyConstr", "Constant"),
        ("GenToken", 1),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", 2),
        ("Reduce",),
        ("Reduce",),
        ("ApplyConstr", "Load"),
    ]
    new_recipe = int2str(recipe)
    assert new_recipe == [
        ("ApplyConstr", "List"),
        ("ApplyConstr", "Constant"),
        ("GenToken", "<int>1"),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", "<int>2"),
        ("Reduce",),
        ("Reduce",),
        ("ApplyConstr", "Load"),
    ]


def test_str2int():
    recipe = [
        ("ApplyConstr", "List"),
        ("ApplyConstr", "Constant"),
        ("GenToken", "<int>1"),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", "<int>2"),
        ("Reduce",),
        ("Reduce",),
        ("ApplyConstr", "Load"),
    ]
    new_recipe = str2int(recipe)
    assert new_recipe == [
        ("ApplyConstr", "List"),
        ("ApplyConstr", "Constant"),
        ("GenToken", 1),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", 2),
        ("Reduce",),
        ("Reduce",),
        ("ApplyConstr", "Load"),
    ]


def test_builder_simple(grammar):
    builder = Builder(grammar)
    recipe = [
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("ApplyConstr", "Store"),
    ]

    builder0 = builder.apply_action(recipe[0])
    assert builder0.get_result() == {"_tag": "Name"}

    builder1 = builder0.apply_action(recipe[1])
    assert builder1.get_result() == {"_tag": "Name", "id": "x"}

    builder2 = builder1.apply_action(recipe[2])
    assert builder2.get_result() == {
        "_tag": "Name",
        "id": "x",
        "ctx": {"_tag": "Store"},
    }


# @pytest.mark.skip(reason="not implemented")
def test_builder_assignment(grammar):
    builder = Builder(grammar)
    recipe = [
        ("ApplyConstr", "Assign"),
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("ApplyConstr", "Store"),
        ("Reduce",),
        ("ApplyConstr", "Constant"),
        ("GenToken", 1),
        ("Reduce",),
        ("Reduce",),
    ]

    builder0 = builder.apply_action(recipe[0])
    assert builder0.get_result() == {
        "_tag": "Assign",
    }

    builder1 = builder0.apply_action(recipe[1])
    assert builder1.get_result() == {
        "_tag": "Assign",
        "targets": [
            {
                "_tag": "Name",
            }
        ],
    }

    builder2 = builder1.apply_action(recipe[2])
    builder3 = builder2.apply_action(recipe[3])
    builder4 = builder3.apply_action(recipe[4])
    assert builder4.get_result() == {
        "_tag": "Assign",
        "targets": [
            {
                "_tag": "Name",
                "id": "x",
                "ctx": {"_tag": "Store"},
            }
        ],
    }

    builder5 = builder4.apply_action(recipe[5])
    assert builder5.get_result() == {
        "_tag": "Assign",
        "targets": [
            {
                "_tag": "Name",
                "id": "x",
                "ctx": {"_tag": "Store"},
            }
        ],
        "value": {
            "_tag": "Constant",
        },
    }

    builder6 = builder5.apply_action(recipe[6])
    builder7 = builder6.apply_action(recipe[7])
    assert not builder7.is_done()

    builder8 = builder7.apply_action(recipe[8])
    assert builder8.get_result() == {
        "_tag": "Assign",
        "targets": [
            {
                "_tag": "Name",
                "id": "x",
                "ctx": {"_tag": "Store"},
            }
        ],
        "value": {
            "_tag": "Constant",
            "value": 1,
            "kind": None,
        },
        "type_comment": None,
    }
    assert builder8.is_done()


def test_builder_empty(grammar):
    builder = Builder(grammar)
    assert builder.is_done() == False
    with pytest.raises(ValueError):
        builder.get_result()


@pytest.mark.skip(reason='Type checking not implemented yet')
def test_builder_wrong_type(grammar):
    builder = Builder(grammar)
    builder.apply_action(('ApplyConstr', 'Expr'))
    with pytest.raises(ValueError):
        builder.apply_action(('GenToken', 'Hello'))


def test_builder_too_many_actions(grammar):
    recipe = [
        ("ApplyConstr", "Constant"),
        ("GenToken", "Hello"),
        ("Reduce",),
    ]
    builder = Builder(grammar)
    for action in recipe:
        builder = builder.apply_action(action)
    with pytest.raises(ValueError):
        builder.apply_action(("GenToken", "World"))
