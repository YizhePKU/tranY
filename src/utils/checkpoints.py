from pathlib import Path


class Checkpoints:
    def __init__(self, dirpath):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def latest(self):
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

    def new(self):
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
