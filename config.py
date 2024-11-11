import yaml

class AttrDict(dict):
    """A dictionary that allows dot notation access to nested keys."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

class Config:
    def __init__(self, config_file='config.yaml', **overrides):
        # Load from YAML file
        self.config = AttrDict(self.load_config(config_file))
        self.apply_overrides(overrides)

    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def apply_overrides(self, overrides):
        for key, value in overrides.items():
            self.set_param(key, value)

    def set_param(self, key, value):
        keys = key.split(".")
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, AttrDict())
        d[keys[-1]] = value

    def apply_overrides_from_file(self, override_file):
        """Apply configuration overrides from another YAML file."""
        overrides = self.load_config(override_file)
        self.apply_overrides_from_dict(overrides)

    def apply_overrides_from_dict(self, overrides, parent_key=''):
        """Apply overrides from a nested dictionary."""
        for key, value in overrides.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                self.apply_overrides_from_dict(value, full_key)
            else:
                self.set_param(full_key, value)

    def __getattr__(self, item):
        return getattr(self.config, item)
