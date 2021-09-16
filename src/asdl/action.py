"""Convert between MR and action sequences.

Currently only DFS-based encoding is implemented(as described in the tranX paper),
but other encodings are also possible.
"""

from collections import OrderedDict, deque
from functools import cache
from copy import deepcopy

import asdl.parser


@cache
def extract_cardinality(grammar):
    """Extract cardinality for all fields in a given grammar.

    Args:
        grammar (asdl.parser.Module): a top-level grammar instance, as returned by `asdl.parser.parse()`.

    Returns:
        (dict[str,OrderedDict[str,str]]): a mapping of tag_name -> field_name -> cardinality
            where cardinality is one of 'single', 'multiple', or 'optional'.
            Field names are listed in declaration order.
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


def mr_to_actions_dfs(mr, grammar):
    """Convert mr to an action sequence in depth-first order, as described in the tranX paper.

    After each field is processed, a "Reduce" action is generated if:
        - a field has cardinality "multiple"
        - a field has cardinality "optional" and currently does not have a value

    Args:
        mr (dict): source mr
        grammar (asdl.parser.Module): reference grammar

    Yields:
        a sequence of actions, where each action can be one of:
            - ("ApplyConstr", tag)
            - ("Reduce",)
            - ("GenToken", token)
    """
    cardinality = extract_cardinality(grammar)
    if isinstance(mr, dict):
        tag_name = mr["_tag"]
        yield ("ApplyConstr", tag_name)
        for field, card in cardinality[tag_name].items():
            if card == "single":
                yield from mr_to_actions_dfs(mr[field], grammar)
            elif card == "multiple":
                for item in mr[field]:
                    yield from mr_to_actions_dfs(item, grammar)
                yield ("Reduce",)
            elif card == "optional":
                if mr[field] is None:
                    yield ("Reduce",)
                else:
                    yield from mr_to_actions_dfs(mr[field], grammar)
            else:
                assert False
    else:
        yield ("GenToken", mr)


def actions_to_mr_dfs(actions, grammar):
    """Convert a depth-first action sequence to mr, as described in the tranX paper.

    Args:
        actions (list[tuple]): a sequence of actions, as returned by ``mr_to_actions_dfs``
        grammar (asdl.parser.Module): reference grammar

    Returns:
        (dict): the reconstructed mr
    """
    cardinality = extract_cardinality(grammar)
    actions = deque(actions)

    def reconstruct_mr():
        # consumes some items from the deque and reconstruct part of the mr
        # returns the reconstructed mr
        frontier = actions.popleft()
        if frontier[0] == "ApplyConstr":
            tag_name = frontier[1]
            retval = dict(_tag=tag_name)
            for field, card in cardinality[tag_name].items():
                if card == "single":
                    retval[field] = reconstruct_mr()
                elif card == "multiple":
                    retval[field] = []
                    while True:
                        if actions[0] == ("Reduce",):
                            actions.popleft()
                            break
                        else:
                            retval[field].append(reconstruct_mr())
                elif card == "optional":
                    if actions[0] == ("Reduce",):
                        actions.popleft()
                        retval[field] = None
                    else:
                        retval[field] = reconstruct_mr()
                else:
                    assert False
            return retval
        elif frontier[0] == "GenToken":
            return frontier[1]
        else:
            assert False

    retval = reconstruct_mr()
    if actions:
        raise Exception("Bad action sequence")
    return retval


def int2str(actions):
    """Replace integers with strings.

    For example, ("GenToken", 1) will be replaced with ("GenToken", "<int>1").

    Args:
        actions (list[tuple]): actions to process.

    Returns:
        (list[tuple]): actions after replacement.
    """
    actions = deepcopy(actions)
    for idx, action in enumerate(actions):
        if action[0] == "GenToken" and isinstance(action[1], int):
            actions[idx] = ("GenToken", f"<int>{action[1]}")
    return actions


def str2int(actions):
    """Restore strings back to integers.

    For example, ("GenToken", "<int>1") will be replaced with ("GenToken", 1).

    Args:
        actions (list[tuple]): actions to process.

    Returns:
        (list[tuple]): actions after replacement.
    """
    actions = deepcopy(actions)
    for idx, action in enumerate(actions):
        if action[0] == "GenToken" and action[1].startswith("<int>"):
            actions[idx] = ("GenToken", int(action[1][5:]))
    return actions
