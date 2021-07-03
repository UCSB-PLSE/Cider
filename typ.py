from abc import ABC, abstractmethod
from typing import Final, Optional

class Type(ABC):
    """Generic type"""
    
    _name: Final[str] # unique name of the type

    @abstractmethod
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name
    
    def __repr__(self) -> str:
        return self._name
    
    @abstractmethod
    def is_base(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_concrete(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def subsume(self, other: "Type") -> "Type":
        """A type T subsumes another type R if
            T is Any(),
            T and R are both Int() or both Bool(), or
            T = Map(T'), R = Map(R') and T' subsumes R'.
        Note that this defines a partial order on types.
        If this type subsumes other, then return
        a substitution (Any() -> S) of T such that
        subst(T) = R.
        """
        raise NotImplementedError

    @abstractmethod
    def subst(self, any_for: "Type") -> "Type":
        """Substitute Any() for any_for"""
        raise NotImplementedError

    def __ge__(self, other):
        """Partial order on types induced by subsume()"""
        return self.subsume(other) is not None
    
    @abstractmethod
    def __eq__(self, other):
        """Equivalence relation on types that models the singleton pattern, i.e.
        Int() == Int(), Map(Bool()) == Map(Bool()), etc."""
        raise NotImplementedError
    
    @abstractmethod
    def __hash__(self):
        return hash(self._name)


class Base(Type):
    """Base types"""
    @abstractmethod
    def __init__(self, name: str):
        super().__init__(name)
    
    def is_base(self):
        return True
    
    def is_concrete(self):
        return True
    
    def __eq__(self, other):
        return isinstance(other, self.__class__)
    
    def __hash__(self):
        return super().__hash__()
    
    def subsume(self, other: Type):
        if isinstance(other, self.__class__):
            return other
        else:
            return None
    
    def subst(self, any_for: Type):
        return self


class Int(Base):
    def __init__(self):
        super().__init__("Int")


class Bool(Base):
    def __init__(self):
        super().__init__("Bool")


class Map(Type):
    _into: Type
    
    def __init__(self, into: Type):
        super().__init__("Map({})".format(str(into)))
        self._into = into

    def destruct(self) -> Type:
        return self._into

    def is_base(self) -> bool:
        return False
    
    def is_concrete(self) -> bool:
        return self._into.is_concrete()
    
    def __eq__(self, other):
        return isinstance(other, Map) and self.destruct() == other.destruct()
    
    def __hash__(self):
        return super().__hash__()
    
    def subsume(self, other):
        if isinstance(other, Map):
            return self.destruct().subsume(other.destruct())
        else:
            return None
    
    def subst(self, any_for: Type):
        return Map(self.destruct().subst(any_for))


class Any(Type):
    def __init__(self):
        super().__init__("Any")
    
    def is_base(self) -> bool:
        return False
    
    def is_concrete(self) -> bool:
        return False
    
    def __eq__(self, other):
        return isinstance(other, Any)
    
    def __hash__(self):
        return super().__hash__()
    
    def subsume(self, other):
        return other
    
    def subst(self, any_for: Type):
        return any_for



if __name__ == "__main__":
    # tests

    # __eq__()
    assert(Int() == Int())
    assert(Bool() == Bool())
    assert(Int() != Bool())
    assert(Bool() != Int())
    assert(Map(Int()) == Map(Int()))
    assert(Map(Bool()) != Map(Int()))
    assert(Any() != Int())
    
    # __ge__() and subsume()
    assert(Any() >= Int())
    assert(Bool() <= Any())
    assert(Map(Int()) <= Map(Any()))
    assert(Map(Map(Int())) <= Map(Any()))
    assert(not Map(Int()) <= Map(Bool()))
    assert(not Map(Int()) >= Map(Bool()))
    assert(Map(Any()).subsume(Map(Map(Bool()))) == Map(Bool()))
    
    # subst()
    assert(Any().subst(Int()) == Int())
    assert(Int().subst(Bool()) == Int())
    assert(Map(Any()).subst(Map(Bool())) == Map(Map(Bool())))
    assert(Map(Map(Int())).subst(Map(Bool())) == Map(Map(Int())))