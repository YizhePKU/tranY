from asdl.transform import transform_mr


def replace_mr(mr, pattern, replacement):
    """Replace the first occurance of `pattern` in `mr` with `replacement`.

    Returns (new_mr, found) where `found` is True when an occurance of `pattern` is found.

    The original MR is left unchanged.
    """
    found = False

    def fn(mr):
        nonlocal found
        if not found and mr == pattern:
            found = True
            return replacement
        else:
            return mr

    new_mr = transform_mr(mr, fn)
    return new_mr, found
