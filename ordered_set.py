from collections.abc import Mapping
from typing import List, Dict, TypeVar, Generic
from abc import ABC, abstractmethod

X = TypeVar("X")


class ORDERED_SET(ABC, Mapping, Generic[X]):
    @abstractmethod
    def index(self, x: X) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def one_hot(self, x: X) -> List[bool]:
        raise NotImplementedError

class OrderedSet(ORDERED_SET, Generic[X]):
    _l: List[X]
    _i: Dict[X, int]
    
    def __init__(self, it):
        self._l = list()
        self._i = dict()
        for i, x in enumerate(it):
            self._l.append(x)
            self._i[x] = i

    def __iter__(self):
        return iter(self._l)
    
    def __getitem__(self, i):
        return self._l[i]
    
    def __len__(self):
        return len(self._l)
    
    def __repr__(self):
        return repr(self._l)
    
    def __str__(self):
        return str(self._l)
    
    def index(self, x: X) -> int:
        return self._i[x]
    
    def one_hot(self, x: X) -> List[bool]:
        i = self.index(x)
        return [j == i for j in range(len(self))]
    

if __name__ == "__main__":
    s = {0, 5, 1, 2, 8, 10, 2, 3, -1}
    o: ORDERED_SET[int] = OrderedSet(s)
    print(", ".join(map(str, s)))
    assert(list(iter(s)) == list(iter(s)))
