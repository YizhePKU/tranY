import ast
import pytest

from asdl.convert import ast_to_mr
import asdl.parser
from asdl.action import extract_cardinality, mr_to_actions_dfs, actions_to_mr_dfs


@pytest.fixture
def grammar():
    return asdl.parser.parse("src/asdl/Python.asdl")


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


def test_mr_to_actions_dfs_variable(grammar):
    mr = {
        "_tag": "Name",
        "id": "x",
        "ctx": {"_tag": "Store"},
    }
    actions = list(mr_to_actions_dfs(mr, grammar))
    assert actions == [
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("ApplyConstr", "Store"),
    ]


def test_mr_to_actions_dfs_assignment(grammar):
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
    actions = list(mr_to_actions_dfs(mr, grammar))
    assert actions == [
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


def test_mr_to_actions_dfs(grammar):
    snippet = "panda.read_csv('file.csv', nrows=100)"
    mr = ast_to_mr(ast.parse(snippet))
    mr = mr["body"][0]  # strip the outer 'module' tag
    actions = list(mr_to_actions_dfs(mr, grammar))
    assert actions == [
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


def test_actions_to_mr_dfs_variable(grammar):
    actions = [
        ("ApplyConstr", "Name"),
        ("GenToken", "x"),
        ("ApplyConstr", "Store"),
    ]
    mr = actions_to_mr_dfs(actions, grammar)
    assert mr == {
        "_tag": "Name",
        "id": "x",
        "ctx": {"_tag": "Store"},
    }


def test_actions_to_mr_dfs_assignment(grammar):
    actions = [
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
    mr = actions_to_mr_dfs(actions, grammar)
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


def test_mr_to_actions_dfs_list(grammar):
    mr = {
        "_tag": "List",
        "elts": [
            {"_tag": "Constant", "value": 1, "kind": None},
            {"_tag": "Constant", "value": 2, "kind": None},
        ],
        "ctx": {"_tag": "Load"},
    }
    actions = list(mr_to_actions_dfs(mr, grammar))
    assert actions == [
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


def test_actions_to_mr_dfs_list(grammar):
    actions = [
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
    mr = actions_to_mr_dfs(actions, grammar)
    assert mr == {
        "_tag": "List",
        "elts": [
            {"_tag": "Constant", "value": 1, "kind": None},
            {"_tag": "Constant", "value": 2, "kind": None},
        ],
        "ctx": {"_tag": "Load"},
    }
