import toml

class Config:
    '''Load a config file in toml and expose values as attributes.'''
    def __init__(self, filepath):
        with open(filepath, 'r') as file:
            self.data = toml.load(file)
        for key, value in self.data.items():
            setattr(self, key, value)
    
    def __str__(self):
        return str(self.data)