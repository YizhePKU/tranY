import pytest

from asdl.replace import replace_mr


def test_replace_mr():
    mr = {
        "_tag": "func",
        "args": [
            {"_tag": "tag1"},
            {"_tag": "tag2"},
        ],
    }

    new_mr, found = replace_mr(mr, {"_tag": "tag2"}, {"_tag": "tag3"})

    assert new_mr == {
        "_tag": "func",
        "args": [
            {"_tag": "tag1"},
            {"_tag": "tag3"},
        ],
    }
    assert found == True
