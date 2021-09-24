import ast

from asdl.convert import ast_to_mr
from asdl.utils import tagged, walk


def test_walk():
    snippet = "x = 1"
    mr = ast_to_mr(ast.parse(snippet))
    for node in walk(mr):
        if tagged(node, "Assign"):
            node["targets"][0]["ctx"]["_tag"] = "Load"
        if tagged(node, "Name"):
            assert node["ctx"]["_tag"] == "Load"
            return
    assert False
