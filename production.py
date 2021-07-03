from typing import Callable, Tuple, Final, TYPE_CHECKING

if TYPE_CHECKING:
    from tree import Node
from typ import Type

class Production:
    """Production of the kind <Type> -> <Node>
    """
    _left: Final[ Type ] # any type, including Any()
    
    _right: Final[ Tuple[Callable[[], "Node"]] ] # a closure that produces some node containing <= one occurrence of Any()
    # note: using Tuple here as a hack, for otherwise mypy would complain about being unable to assign to _right in __init__
    
    _name: Final[ str ] # uniquely identifies a production

    def __init__(self, left: Type, right: Callable[[], "Node"], name: str):
        self._left = left
        self._right = (right,)
        self._name = name
    
    def is_concrete(self):
        return self._left.is_concrete()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def left(self) -> Type:
        return self._left

    @property
    def right(self) -> Callable[[], "Node"]:
        return self._right[0]

    def hash(self):
        return hash(self._name)

    def __repr__(self):
        return self._name
    
    def __str__(self):
        return "\"{}\": ?{} --> {}".format(self.name, str(self.left), str(self.right()))