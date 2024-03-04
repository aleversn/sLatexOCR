import json

class JsonData:
    def __init__(self, load_path=None):
        if load_path is not None:
            self.load(load_path)
        self.__dict__ = self.data

    def load(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)
    
    def get(self, key, default=None):
        if key not in self.data:
            return default
        return self.data[key]
