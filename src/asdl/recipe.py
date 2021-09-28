"""Provides tools for working with recipes.

A recipe is a list of actions used to construct a MR. Each action is one of:
  - ("ApplyConstr", tag)
  - ("Reduce",)
  - ("GenToken", token)

Why the name "recipe"? We could just refer to them as "actions", but then it's hard to
find a good plural form ("action's"?). The original tranX implementation uses "hypothesis"
and "hypotheses", but that's way too hard to spell correctly.

Currently only DFS-based encoding is implemented (as described in the tranX paper),
but other encodings are also possible.
"""

from collections import OrderedDict, deque, namedtuple
from copy import deepcopy
from functools import cache

from pyrsistent import get_in, m, pmap, pvector, v

import asdl.parser


@cache
def extract_cardinality(grammar):
    """Extract cardinality for all fields in a given grammar.

    Args:
        grammar: a top-level grammar instance, as returned by `asdl.parser.parse()`.

    Returns:
        a mapping of tag_name -> field_name -> cardinality, where cardinality is
        one of 'single', 'multiple', or 'optional'. Field names are listed in
        declaration order.
    """
    retval = {}

    def handle_constructor(constructor, tag_name):
        retval[tag_name] = OrderedDict()
        for field in constructor.fields:
            field_name = field.name
            if field.seq:
                cardinality = "multiple"
            elif field.opt:
                cardinality = "optional"
            else:
                cardinality = "single"
            retval[tag_name][field_name] = cardinality

    for type, value in grammar.types.items():
        if isinstance(value, asdl.parser.Sum):
            for constructor in value.types:
                handle_constructor(constructor, tag_name=constructor.name)
        elif isinstance(value, asdl.parser.Product):
            handle_constructor(value, tag_name=type)
        else:
            assert False

    return retval


@cache
def preprocess_grammar(grammar):
    """Convert a grammar instance into plain data.

    Args:
        grammar: a grammar instance.

    Returns:
        type2constr (dict[str,list[str]]): list constructors of a given type.
        constr2type (dict[str,str]): lookup the type of a constructor.
        fields (dict[str,list[tuple[str,str,str]]]): list fields of a constructor
            in the order they are declared in the grammar, where each field
            is a namedtuple of (type, name, cardinality). cardinality can be one of
            'single', 'multiple', or 'optional'.
    """
    Field = namedtuple("Field", ["type", "name", "cardinality"])
    type2constr = {}
    constr2type = {}
    fields = {}

    def _handle_constr(type, name, obj):
        type2constr[type].append(name)
        constr2type[name] = type
        fields[name] = []
        for field in obj.fields:
            if field.seq:
                cardinality = "multiple"
            elif field.opt:
                cardinality = "optional"
            else:
                cardinality = "single"
            fields[name].append(Field(field.type, field.name, cardinality))

    for type, value in grammar.types.items():
        type2constr[type] = []
        if isinstance(value, asdl.parser.Sum):
            for constr in value.types:
                _handle_constr(type, constr.name, constr)
        elif isinstance(value, asdl.parser.Product):
            _handle_constr(type, type, value)
        else:
            assert False
    return type2constr, constr2type, fields


def _generator_to_list_function(f):
    def _inner(*args, **kwargs):
        return list(f(*args, **kwargs))

    return _inner


@_generator_to_list_function
def mr_to_recipe_dfs(mr, grammar):
    """Convert MR to a recipe in depth-first order, as described in the tranX paper.

    After each field is processed, a "reduce" action is generated if:
        - a field has cardinality "multiple"
        - a field has cardinality "optional" and currently does not have a value

    Args:
        mr: MR to convert from
        grammar: reference grammar

    Returns:
        a recipe, the conversion result
    """
    cardinality = extract_cardinality(grammar)
    if isinstance(mr, dict):
        tag_name = mr["_tag"]
        yield ("ApplyConstr", tag_name)
        for field, card in cardinality[tag_name].items():
            if card == "single":
                yield from mr_to_recipe_dfs(mr[field], grammar)
            elif card == "multiple":
                for item in mr[field]:
                    yield from mr_to_recipe_dfs(item, grammar)
                yield ("Reduce",)
            elif card == "optional":
                if mr[field] is None:
                    yield ("Reduce",)
                else:
                    yield from mr_to_recipe_dfs(mr[field], grammar)
            else:
                assert False
    else:
        yield ("GenToken", mr)


def recipe_to_mr_dfs(recipe, grammar):
    """Convert a depth-first recipe to MR, as described in the tranX paper.

    Args:
        recipe: recipe to convert from
        grammar: reference grammar

    Returns:
        the reconstructed MR

    Raises:
        ValueError: when the recipe does not conform to grammar.
    """
    cardinality = extract_cardinality(grammar)
    recipe = deque(recipe)

    def reconstruct_mr():
        # consumes some items from the deque and reconstruct part of the MR
        # returns the reconstructed MR
        frontier = recipe.popleft()
        if frontier[0] == "ApplyConstr":
            tag_name = frontier[1]
            retval = dict(_tag=tag_name)
            for field, card in cardinality[tag_name].items():
                if card == "single":
                    retval[field] = reconstruct_mr()
                elif card == "multiple":
                    retval[field] = []
                    while True:
                        if recipe[0] == ("Reduce",):
                            recipe.popleft()
                            break
                        else:
                            retval[field].append(reconstruct_mr())
                elif card == "optional":
                    if recipe[0] == ("Reduce",):
                        recipe.popleft()
                        retval[field] = None
                    else:
                        retval[field] = reconstruct_mr()
                else:
                    assert False
            return retval
        elif frontier[0] == "GenToken":
            return frontier[1]
        else:
            raise ValueError("Bad recipe")

    retval = reconstruct_mr()
    if recipe:
        raise ValueError("Bad recipe")
    return retval


def int2str(recipe):
    """Replace integers in a recipe with strings.

    For example, ("GenToken", 1) will be replaced with ("GenToken", "<int>1").

    Args:
        recipe: recipe to process.

    Returns:
        a recipe after the replacement.
    """
    recipe = deepcopy(recipe)
    for idx, action in enumerate(recipe):
        if action[0] == "GenToken" and isinstance(action[1], int):
            recipe[idx] = ("GenToken", f"<int>{action[1]}")
    return recipe


def str2int(recipe):
    """Restore strings in a recipe back to integers.

    For example, ("GenToken", "<int>1") will be replaced with ("GenToken", 1).

    Args:
        recipe: recipe to process.

    Returns:
        a recipe after being restored.
    """
    recipe = deepcopy(recipe)
    for idx, action in enumerate(recipe):
        if action[0] == "GenToken" and action[1].startswith("<int>"):
            recipe[idx] = ("GenToken", int(action[1][5:]))
    return recipe


class Builder:
    """A builder constructs a MR from a recipe, step by step."""

    def __init__(self, grammar):
        self._grammar = grammar
        self._result = m(_tag="ROOT")
        self._stack = v(v("toplevel"))

    def _copywith(self, result, stack):
        builder = Builder(self._grammar)
        builder._result = result
        builder._stack = stack
        return builder

    def get_result(self):
        if "toplevel" in self._result:
            return self._result["toplevel"]
        else:
            raise ValueError("Cannot get the result of an empty builder")

    def is_done(self):
        return len(self._stack) == 0

    def apply_action(self, action):
        _, _, fields = preprocess_grammar(self._grammar)
        stack = self._stack
        result = self._result

        # pop a path from the stack
        *stack, path = stack

        # if we're adding values to a list, add another path to keep adding to the list
        if isinstance(path[-1], int) and action[0] != "Reduce":
            stack.append(path.set(-1, path[-1] + 1))

        # make an empty list when opening a list
        if path[-1] == 0:
            result = result.transform(path[:-1], v())

        if action[0] == "ApplyConstr":
            # open a constructor
            result = result.transform(path, m(_tag=action[1]))
            # add paths for the fields of the constructor (in reverse order)
            for field in reversed(fields[action[1]]):
                if field.cardinality == "multiple":
                    stack.append(path + [field.name, 0])
                else:
                    stack.append(path + [field.name])
            return self._copywith(result, stack)
        elif action[0] == "Reduce":
            if isinstance(path[-1], int):
                # close the list (no-op)
                return self._copywith(result, stack)
            else:
                # set None to an optional field
                # but check if the field is really optional first
                parent = get_in(path[:-1], result)
                field = next(
                    field for field in fields[parent["_tag"]] if field.name == path[-1]
                )
                if field.cardinality == "optional":
                    result = result.transform(path, None)
                    return self._copywith(result, stack)
                else:
                    raise ValueError("Unexpected Reduce")
        elif action[0] == "GenToken":
            # add a token to the tree
            result = result.transform(path, action[1])
            return self._copywith(result, stack)
        else:
            raise ValueError("Unknown action")
