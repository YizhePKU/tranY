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

from collections import deque, namedtuple
from copy import deepcopy
from functools import cache

import asdl.parser

# Represents a field of a constructor.
# "cardinality" is one of "single", "multiple", or "optional"
Field = namedtuple("Field", ["type", "name", "cardinality"])


@cache
def preprocess_grammar(grammar):
    """Convert a grammar instance into plain data.

    A fake grammar rule "root = (mod toplevel)" is added for easy processing.

    Args:
        grammar: a grammar instance.

    Returns:
        type2constr (dict[str,list[str]]): a list of names of constructors of a given type.
        constr2type (dict[str,str]): name of the type of a constructor.
        fields (dict[str,list[Field]]): a list of fields of a constructor in declaration order.
        name2field (dict[str,dict[str,Field]]): a dict of fields of a constructor, keyed by field name.
    """
    type2constr = {}
    constr2type = {}
    fields = {}
    name2field = {}

    # add the fake grammar rule
    type2constr["root"] = ["root"]
    constr2type["root"] = "root"
    fields["root"] = [Field("mod", "toplevel", "single")]
    name2field["root"] = {"toplevel": Field("mod", "toplevel", "single")}

    def _handle_constr(type, name, obj):
        type2constr[type].append(name)
        constr2type[name] = type
        fields[name] = []
        name2field[name] = {}
        for field in obj.fields:
            if field.seq:
                cardinality = "multiple"
            elif field.opt:
                cardinality = "optional"
            else:
                cardinality = "single"
            fields[name].append(Field(field.type, field.name, cardinality))
            name2field[name][field.name] = Field(field.type, field.name, cardinality)

    for type, value in grammar.types.items():
        type2constr[type] = []
        if isinstance(value, asdl.parser.Sum):
            for constr in value.types:
                _handle_constr(type, constr.name, constr)
        elif isinstance(value, asdl.parser.Product):
            _handle_constr(type, type, value)
        else:
            assert False
    return type2constr, constr2type, fields, name2field


def _generator_to_list_decorator(fn):
    """Convert a generator function to a function that returns a list.

    This enables us to use the Python `yield` syntax to create a list.
    """

    def inner(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return inner


@_generator_to_list_decorator
def mr_to_recipe_dfs(mr, grammar):
    """Convert MR to a recipe in depth-first order, as described in the tranX paper.

    After each field is processed, a "reduce" action is generated if:
        - a field has cardinality "multiple"
        - a field has cardinality "optional" and currently does not have a value

    Args:
        mr: MR to convert from
        grammar: an ASDL grammar instance

    Returns:
        a recipe, the conversion result
    """
    _, _, fields, _ = preprocess_grammar(grammar)
    if isinstance(mr, dict):
        tag_name = mr["_tag"]
        yield ("ApplyConstr", tag_name)
        for _, field, card in fields[tag_name]:
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
        grammar: an ASDL grammar instance

    Returns:
        the reconstructed MR

    Raises:
        ValueError: when the recipe does not conform to grammar.
    """
    _, _, fields, _ = preprocess_grammar(grammar)
    recipe = deque(recipe)

    def reconstruct_mr():
        # consumes some items from the deque and reconstruct part of the MR
        # returns the reconstructed MR
        frontier = recipe.popleft()
        if frontier[0] == "ApplyConstr":
            tag_name = frontier[1]
            retval = dict(_tag=tag_name)
            for _, field, card in fields[tag_name]:
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


class Builder:
    """A builder constructs a MR from a recipe, step by step.

    Note that a builder is immutable; calling apply_action() returns a new builder.
    """

    def __init__(self, grammar):
        self._grammar = grammar
        self._result = m(_tag="root")
        # self._stack is a stack of paths, where each path is a list of keys
        # pointing to the field being modified.
        #
        # This is used to emulate recursive calls.
        #
        # For example, if the last value of self._stack is
        #     ["toplevel", "body", "args", 2, "value"]
        # then the next field to modify is
        #     self._result["toplevel"]["body"]["args"][2]["value"]
        #
        # Note that if the cardinality of the field is multiple,
        # the path ends with the field name, NOT an index.
        self._stack = v(v("toplevel"))
        self._history = v()

    def _copywith(self, result, stack, history):
        builder = Builder(self._grammar)
        builder._result = result
        builder._stack = stack
        builder._history = history
        return builder

    @property
    def result(self):
        """Return the constructed MR so far, or None if the builder is empty."""
        if "toplevel" in self._result:
            return self._result["toplevel"]
        else:
            return None

    @property
    def done(self):
        """Return whether the MR being built is complete."""
        return len(self._stack) == 0

    @property
    def current_frontier(self):
        """Return the name of the frontier constructor."""
        if self._stack:
            return get_in(self._stack[-1][:-1], self._result)["_tag"]
        else:
            raise ValueError("Cannot get current frontier of a completed builder.")

    @property
    def current_field(self):
        """Return the field (type, name, cardinality) to be updated by the next action."""
        if self._stack:
            _, _, _, name2field = preprocess_grammar(self._grammar)
            path = self._stack[-1]
            return name2field[self.current_frontier][path[-1]]
        else:
            raise ValueError("Cannot get current field of a completed builder.")

    @property
    def allowed_actions(self):
        """Return a list of actions that are OK to apply.

        When the cardinality of the next field is "multiple", the result
        includes ("Reduce",)

        When the type of the next field is an identifier, integer, or string, the result
        includes ("GenToken",). It's not a real action, of course, just a placeholder
        for any GenToken action.
        """
        type2constr, _, _, _ = preprocess_grammar(self._grammar)
        field_type, _, field_cardinality = self.current_field
        if field_type in ("identifier", "int", "string", "constant"):
            actions = [("GenToken",)]
        else:
            actions = [("ApplyConstr", constr) for constr in type2constr[field_type]]
        if field_cardinality in ["multiple", "optional"]:
            actions.append(("Reduce",))
        return actions

    @property
    def history(self):
        return thaw(self._history)

    def _is_action_allowed(self, action):
        allowed_actions = self.allowed_actions
        if action in allowed_actions:
            return True
        if action[0] == "GenToken" and ("GenToken",) in allowed_actions:
            return True
        return False

    def apply_action(self, action):
        """Apply an action to the MR.

        Returns:
            a builder with updated MR.

        Raises:
            ValueError if the given action is illegal to apply.
                (type errors, grammar errors, etc.)
        """
        if not self._stack:
            raise ValueError("Cannot apply action to a completed builder")

        if not self._is_action_allowed(action):
            raise ValueError("Trying to apply an illegal action")

        _, _, fields, _ = preprocess_grammar(self._grammar)
        stack = self._stack
        result = self._result
        history = self._history.append(action)

        # pop a path from the stack
        stack, path = stack[:-1], stack[-1]

        if self.current_field.cardinality == "multiple":
            # create the list if necessary
            if not get_in(path, result):
                result = result.transform(path, v())
            # if we're not closing the list, push this path back onto the stack
            if action[0] != "Reduce":
                stack = stack.append(path)
            # also, append an index to path so that we can add items to the list
            path = path.append(len(get_in(path, result)))

        if action[0] == "ApplyConstr":
            # open a constructor
            result = result.transform(path, m(_tag=action[1]))
            # add paths for the fields of the constructor (in reverse order)
            for field in reversed(fields[action[1]]):
                stack = stack.append(path + [field.name])
            return self._copywith(result, stack, history)
        elif action[0] == "Reduce":
            cardinality = self.current_field.cardinality
            if cardinality == "multiple":
                # close the list (no-op)
                return self._copywith(result, stack, history)
            elif cardinality == "optional":
                # set None to an optional field
                result = result.transform(path, None)
                return self._copywith(result, stack, history)
            else:
                assert False
        elif action[0] == "GenToken":
            # add a token to the tree
            result = result.transform(path, action[1])
            return self._copywith(result, stack, history)
        else:
            assert False
