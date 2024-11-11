import yaml

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

class Config:
    def __init__(self, config_file='config.yaml', **overrides):
        self.config = load_config(config_file)
        self.apply_overrides(overrides)

    def apply_overrides(self, overrides):
        for key, value in overrides.items():
            self.set_param(key, value)

    def set_param(self, key, value):
        keys = key.split(".")
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
