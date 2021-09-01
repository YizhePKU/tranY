import pytest

from asdl.transform import transform_mr


def test_transform():
    mr = {
        "_tag": "func",
        "args": [
            {"_tag": "tag1"},
            {"_tag": "tag2"},
        ],
    }

    def fn(mr):
        if mr == {"_tag": "tag2"}:
            return {"_tag": "tag3"}
        else:
            return mr

    new_mr = transform_mr(mr, fn)
    
    assert new_mr == {
        "_tag": "func",
        "args": [
            {"_tag": "tag1"},
            {"_tag": "tag3"},
        ],
    }
