from typing import Union, Optional
from pathlib import Path


class Checkpoints:
    """Class for saving and loading latest checkpoints.

    A checkpoint is a file named `{version}.pt` inside a specific directory.
    `latest()` gives you a path to the latest checkpoint, and `new()` gives you
    a path to a (non-existing) checkpoint that is newer than existing ones.

    For example, suppose we have a directory structure like this:

    - models
      - model_v1
        - 1.pt
        - 2.pt
      - model_v2
        - 1.pt
        - 2.pt
        - 3.pt

    Then we have:

    ```
    >>> Checkpoints('models/model_v1').latest()
    'models/model_v1/2.pt'

    >>> Checkpoints('models/model_v1').new()
    'models/model_v1/3.pt'
    ```

    `dirpath` needs not exists; it will be created if necessary.

    This class is NOT thread-safe. Race conditions could occur if multiple
    instances of this class are created for the same directory.
    """

    def __init__(self, dirpath: Union[str, Path]):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def latest(self) -> Optional[Path]:
        """Returns path to the latest checkpoint, or None if no checkpoint exists."""
        latest_version = 0
        for filepath in self.dirpath.iterdir():
            if filepath.is_file():
                try:
                    version = int(filepath.stem)
                except:
                    pass
                if version > latest_version:
                    latest_version = version
                    latest_filepath = filepath
        if latest_version > 0:
            return self.dirpath / latest_filepath.name
        else:
            return None

    def new(self) -> Path:
        """Returns path to a new, non-existing checkpoint."""
        latest_version = 0
        for filepath in self.dirpath.iterdir():
            if filepath.is_file():
                try:
                    version = int(filepath.stem)
                except:
                    pass
                if version > latest_version:
                    latest_version = version
        return self.dirpath / f"{latest_version + 1}.pt"
