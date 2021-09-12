"""Convert between Python AST and MR.

An MR node is either a dictionary or a constant(including None). Lists are not considered MR.
The `Python.asdl` file serves as a schema for the MR.

Quote from https://docs.python.org/3/library/ast.html#ast.AST._fields
If the attributes are optional (marked with a question mark), the value might be None.
If the attributes can have zero-or-more values (marked with an asterisk), the values are represented as Python lists.
"""

import ast


def ast_to_mr(root):
    """Convert Python AST to MR."""
    if isinstance(root, ast.AST):
        mr = {"_tag": root.__class__.__name__}
        for field in root._fields:
            value = getattr(root, field)
            if value is None:
                mr[field] = None
            elif isinstance(value, list):
                mr[field] = [ast_to_mr(item) for item in value]
            else:
                mr[field] = ast_to_mr(value)
        return mr
    else:
        return root


def mr_to_ast(mr):
    """Convert MR to Python AST."""
    if isinstance(mr, dict):
        constructor = getattr(ast, mr["_tag"])
        kwargs = {}
        for field, value in mr.items():
            if field != "_tag":
                kwargs[field] = mr_to_ast(value)
        return constructor(**kwargs)
    else:
        return mr
