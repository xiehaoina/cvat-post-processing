__author__ = 'xiehaoina'
from collections import abc

class json_attr:
    def __new__(cls, arg):
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg,abc.MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def __init__(self, mapping):
        self.__data = {}
        for key, value in mapping.items():
            if key.iskeyword(key):
                key += '_'
            self.__data[key] = value