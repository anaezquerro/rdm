from __future__ import annotations
import os
from ruamel.yaml import YAML

class Config:
    def __init__(self, **kwargs):
        for field, value in kwargs.items():
            self.__setattr__(field, value)
            
    def __iter__(self):
        for field in self.keys():
            yield field, self.__getattribute__(field)
                
    def __len__(self):
        return len(self.keys())
    
    def __getattr__(self, field: str):
        return None
        
    def __getitem__(self, field: str):
        return self.__getattribute__(field)
    
    def copy(self) -> Config:
        return Config(**self)
    
    def keys(self) -> list[str]:
        return [field for field in self.__dict__ if not field.startswith('_')]

    def __repr__(self) -> str:
        return f'Config(' + ', '.join(f'{name}={value}' for name, value in self) + ')'
            
    @classmethod
    def from_yaml(cls, path: str) -> Config:
        assert os.path.exists(path), f'Configuration file {path} odes not exist'
        yaml = YAML(typ='safe')
        config = yaml.load(open(path, 'r'))
        return cls.from_dict(config)
        
    @classmethod
    def from_dict(cls, data):
        values = dict()
        for name, value in data.items():
            if isinstance(value, dict):
                values[name] = cls.from_dict(value)
            else:
                values[name] = value 
        return cls(**values)