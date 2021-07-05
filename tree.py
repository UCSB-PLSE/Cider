from abc import ABC, abstractmethod
from typing import List, Set, Optional
from copy import deepcopy

from typ import Type
from production import Production
from graph import MyGraph


class Node(ABC):
    _type: Type
    _children: List["Node"]

    @abstractmethod
    def __init__(self, typ: Type, children: List["Node"]):
        self._type = typ
        self._children = children
    
    @abstractmethod
    def is_complete(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError
    
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
    
    def tokenize(self) -> List[str]:
        res: List[str] = []
        self._tokenize(res)
        return res
    
    @abstractmethod
    def _tokenize(self, acc: List[str]):
        raise NotImplementedError
    
    @property
    def type(self) -> Type:
        return self._type
    
    @abstractmethod
    def _expand(self, p: Production) -> Optional["Node"]:
        raise NotImplementedError
    
    def expand(self, p: Production) -> "Node":
        expanded = self._expand(p)
        assert(expanded is not None)
        return expanded
    
    @abstractmethod
    def holes(self) -> Set[Type]:
        raise NotImplementedError
    
    @abstractmethod
    def nonholes(self) -> Set[str]:
        raise NotImplementedError

    @abstractmethod
    def subst(self, any_for: Type):
        raise NotImplementedError
    
    def types(self):
        return set([self.type]).union(*[c.types() for c in self._children])

    def to_graph(self):
        self_copy = deepcopy(self)
        special_nodes = {t: str(t) for t in self_copy.types()}
        
        def get_count():
            res = get_count.count
            get_count.count += 1
            return res
        get_count.count = 0
        
        def assign_id(node):
            node.i = str(get_count())
            for c in node._children:
                assign_id(c)
        assign_id(self_copy)

        g = MyGraph()
        def f(node):
            g.add_undirected(node.i, special_nodes[node.type])
            for c in node._children:
                g.add_undirected(node.i, c.i)
                f(c)
        f(self_copy)
        return g
    

class Hole(Node):
    def __init__(self, typ: Type):
        super().__init__(typ, [])
    
    def __str__(self):
        return f"?{str(self._type)}"
    
    def __repr__(self):
        return f"?{repr(self._type)}"
    
    def _expand(self, p: Production) -> Optional[Node]:
        sigma = p.left.subsume(self.type)
        if sigma:
            res = p.right()
            res.subst(sigma)
            return res
        else:
            return None
    
    def subst(self, any_for: Type):
        self._type = self._type.subst(any_for)
    
    def is_complete(self) -> bool:
        return False
    
    def holes(self) -> Set[Type]:
        return {self.type}
    
    def nonholes(self) -> Set[str]:
        return set()
    
    def _tokenize(self, acc: List[str]):
        acc.append(str(self.type))


class NonHole(Node):
    _kind: str

    @abstractmethod
    def __init__(self, kind: str, typ: Type, children: List["Node"]):
        super().__init__(typ, children)
        self._kind = kind
    
    def __repr__(self):
        return "{op}: {typ} [{children}]".format(
            op=self._kind,
            typ=repr(self._type),
            children=", ".join([repr(c) for c in self._children]))
    
    @property
    def kind(self):
        return self._kind
    
    def _expand(self, p: Production) -> Optional[Node]:
        """Perform left-most expansion"""
        for i, c in enumerate(self._children):
            c_expanded = c._expand(p)
            if c_expanded:
                self._children[i] = c_expanded
                return self
        return None
    
    def subst(self, any_for: Type):
        for c in self._children:
            c.subst(any_for)
    
    def is_complete(self) -> bool:
        return all(c.is_complete() for c in self._children)
    
    def holes(self) -> Set[Type]:
        res = set()
        for c in self._children:
            res |= c.holes()
        return res
    
    def nonholes(self) -> Set[str]:
        res = {self.kind}
        for c in self._children:
            res |= c.nonholes()
        return res
    
    def _tokenize(self, acc: List[str]):
        acc.append("(")
        acc.append(self.kind)
        for c in self._children:
            c._tokenize(acc)
        acc.append(")")


class Unary(NonHole):
    def __init__(self, kind: str, typ: Type, child: Node):
        super().__init__(kind, typ, [child])
    
    def __str__(self):
        return "{op}({c})".format(op=self._kind, c=self._children[0])



class Infix(NonHole):
    def __init__(self, kind: str, typ: Type, left: Node, right: Node):
        super().__init__(kind, typ, [left, right])
    
    def __str__(self):
        return "{l} {o} {r}".format(l=self._children[0], o=self._kind, r=self._children[1])


class Leaf(NonHole):
    def __init__(self, kind: str, typ: Type):
        super().__init__(kind, typ, [])
    
    def __str__(self):
        return self.kind


if __name__ == '__main__':
    from typ import Int, Bool, Map, Any
    from production import Production
    nodes = {
        "i": lambda: Hole(Int()),
        "mi": lambda: Hole(Map(Int())),
        "mma": lambda: Hole(Map(Map(Any()))),
        "0": lambda: Leaf("0", Int()),
        "<=": lambda: Infix("<=", Bool(), Hole(Int()), Hole(Int())),
        "sum": lambda: Unary("sum", Int(), Leaf("bal", Map(Int()))),
        "+": lambda: Infix("+", Int(), Hole(Int()), Hole(Int()))
    }
    ps = {
        "0": Production(Int(), nodes["0"], "0"),
        "<=": Production(Bool(), nodes["<="], "<="),
        "+": Production(Int(), nodes["+"], "+"),
        "flatten": Production(Map(Any()), nodes["mma"], "flatten"),
        "sum": Production(Int(), nodes["mi"], "sum"),
    }

    h: Node = Hole(Bool())
    print(h)
    seq = ["<=", "sum", "flatten", "+", "0", "0"]
    holes = [{Int()}, {Map(Int()), Int()}, {Map(Map(Int())), Int()}, {Map(Map(Int())), Int()}, {Map(Map(Int())), Int()}, {Map(Map(Int()))}]
    for i, s in enumerate(seq):
        p = ps[s]
        print(s, end=" :: ")
        h1 = h.expand(p)
        assert(h1 is not None)
        h = h1
        print(h)
        if h.holes() != holes[i]:
            print(h.holes())
            print(holes[i])
            assert(False)
    
    print(h.to_graph())