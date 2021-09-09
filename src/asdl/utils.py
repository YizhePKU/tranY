def tagged(mr, tag):
    """Check if the MR has a given tag."""
    return isinstance(mr, dict) and mr["_tag"] == tag


def walk(mr):
    """Recursively yield all descendant MR, starting with itself, in no specified order.

    This is useful if you only want to modify nodes in place and don’t care about the context.
    """
    yield mr
    if isinstance(mr, dict):
        for field, body in mr.items():
            if isinstance(body, list):
                for item in body:
                    yield from walk(item)
            else:
                yield from walk(body)
