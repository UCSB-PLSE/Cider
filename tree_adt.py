from dataclasses import dataclass
from typing import TypeVar, Union, NoReturn, List
from itertools import chain

S = TypeVar("a")
@dataclass()
class Node:
    children: List['Node']

@dataclass
class Leaf:
    val: S

Tree = Union[Node, Leaf]

def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))


def visit(t: Tree) -> List[S]:
    if isinstance(t, Node):
        return list(chain.from_iterable(visit(c) for c in t.children))
    elif isinstance(t, Leaf):
        return [t.val]
    else:
        assert_never(t)


t = Node([
        Node([
            Leaf(1)]),
    Leaf('a')])

