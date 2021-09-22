import pytest
from pathlib import Path

from utils.checkpoints import Checkpoints


@pytest.fixture
def checkpoints(tmp_path):
    return Checkpoints(tmp_path)


def test_new_fresh_dir(tmp_path):
    dirpath = Path(tmp_path)
    path1 = Checkpoints(dirpath / "hello" / "world").new()
    assert path1 == dirpath / "hello" / "world" / "1.pt"


def test_new_once(checkpoints):
    path1 = checkpoints.new()
    assert path1 == checkpoints.dirpath / "1.pt"


def test_new_twice(checkpoints):
    path1 = checkpoints.new()
    path1.touch()
    path2 = checkpoints.new()
    assert path2 == checkpoints.dirpath / "2.pt"


def test_latest_empty(checkpoints):
    assert checkpoints.latest() == None


def test_latest_new_twice(checkpoints):
    path1 = checkpoints.new()
    path1.touch()
    path1a = checkpoints.latest()
    assert path1 == path1a

    path2 = checkpoints.new()
    path2.touch()
    path2a = checkpoints.latest()
    assert path2 == path2a

    assert path1a != path2a
