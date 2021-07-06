from collections.abc import Mapping
from typing import List, Tuple, Dict, Callable, Final, TypeVar, Generic
from abc import ABC, abstractmethod

K = TypeVar("K")
V = TypeVar("V")


class ORDERED_DICT(ABC, Mapping, Generic[K,V]):
    @abstractmethod
    def nth(self, i: int) -> Tuple[K,V]:
        raise NotImplementedError
    
    @abstractmethod
    def index(self, k: K) -> int:
        raise NotImplementedError


class OrderedDict(ORDERED_DICT, Generic[K,V]):
    _l: List[Tuple[K,V]]
    _d: Dict[K, V]
    _i: Dict[K, int]

    def __init__(self, kv_lst: List[Tuple[K,V]]):
        self._l = kv_lst
        self._d = {k: v for k,v in kv_lst}
        self._i = {k: i for i, (k,_) in enumerate(kv_lst)}
    
    def __getitem__(self, k: K) -> V:
        return self._d[k]
    
    def nth(self, i: int) -> Tuple[K,V]:
        return self._l[i]
    
    def __iter__(self):
        return iter(self._l)
    
    def index(self, k: K) -> int:
        return self._i[k]
    
    def __len__(self):
        return len(self._l)

if __name__ == "__main__":
    ks = [5,4,3,7,8,6,0,1,2]
    kv_lst = [(k, str(k)) for k in ks]
    d = OrderedDict(kv_lst)
    
    # test nth
    for i in range(len(ks)):
        assert(d.nth(i) == kv_lst[i])
    
    # test iter
    for i, kv in enumerate(d):
        assert(kv == kv_lst[i])
    
    # test index, __getitem__
    import random
    ks_copy = ks[::]
    random.shuffle(ks_copy)
    for k in ks_copy:
        assert(d.index(k) == ks.index(k))
        assert(d[k] == str(k))