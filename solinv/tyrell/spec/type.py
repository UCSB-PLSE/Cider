from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any
from .expr import ExprType
from .sort import Sort, ANY

class Type(ABC):
    '''A generic class for types in DSL'''
    _name: str
    _sort: Optional[Sort]

    @abstractmethod
    def __init__(self, name: str, sort: Optional[Sort] = None):
        self._name = name
        self._sort = sort

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def sort(self) -> Optional[Sort]:
        return self._sort

    @abstractmethod
    def is_enum(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_value(self) -> bool:
        raise NotImplementedError

    def __str__(self) -> str:
        if self._sort:
            return f"{self._name} <{self._sort}>"
        else:
            return self._name

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Type):
            return self._name == other._name and self._sort == other._sort
        return NotImplemented
    
    def subsume(self, other):
        if isinstance(other, Type):
            if self._name == other._name:
                return self._sort.subsume(other._sort)
            else:
                return None
        return NotImplemented
    
    def __ge__(self, other):
        return self.subsume(other) is not None
    
    def subst(self, any_for: Sort) -> 'Type':
        this = deepcopy(self)
        this._sort = this._sort.subst(any_for)
        return this

    def __hash__(self):
        """Overrides the default implementation"""
        return hash((self._name, self._sort))


class EnumType(Type):
    '''A special kind of type whose domain is finite and specified up-front'''

    _domain: List[Any]

    def __init__(self, name: str, domain: List[Any] = []):
        super().__init__(name, ANY)
        self._domain = domain

    @property
    def domain(self) -> List[Any]:
        return self._domain

    def is_enum(self) -> bool:
        return True

    def is_value(self) -> bool:
        return False

    def __repr__(self) -> str:
        return 'EnumType({}, domain={})'.format(self._name, self._domain)


class ValueType(Type):
    _properties: Dict[str, ExprType]

    def __init__(self, name: str, properties: List[Tuple[str, ExprType]] = [], sort: Optional[Sort] = None):
        super().__init__(name, sort)
        self._properties = dict()
        for name, ty in properties:
            if name in self._properties:
                raise ValueError('Duplicate property name: {}'.format(name))
            self._properties[name] = ty

    def is_enum(self) -> bool:
        return False

    def is_value(self) -> bool:
        return True

    def get_property(self, name: str) -> Optional[ExprType]:
        return self._properties.get(name, None)

    def get_property_or_raise(self, name: str) -> ExprType:
        return self._properties[name]

    @property
    def properties(self):
        return list(self._properties.items())

    def __repr__(self) -> str:
        return 'ValueType({}, properties={})'.format(self._name, self._properties)
