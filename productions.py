from typing import Dict, List, Set, Final
from collections.abc import Mapping
from production import Production
from typ import Type, Bool, Int, Map, Any
from tree import Hole, Infix, Unary, Leaf

class Productions(Mapping):
    _d: Final[ Dict[str, Production] ]
    _l: Final[ List[Production] ]
    _i: Final[ Dict[Production, int] ]
    _accessible: Final[ Dict[Type, Set[Production]] ]
    
    def __init__(self, p_lst: List[Production], order=None):
        self._l = p_lst
        self._d = {p.name: p for p in p_lst}
        self._i = {p: i for i,p in enumerate(p_lst)}
        self._accessible = self.compute_accessible()
    
    def compute_accessible(self):
        """Map concrete types to the set of productions that can (eventually)
        produce terms of this type
        
        Inductively, a concrete type t is accessible if there is a production
        p: t ==> right such that either
        (1) right is complete (no holes), or
        (2) right is incomplete, and
            - all concrete type holes are accessible
            - the only abstract type hole can be instantited by some accessible type
        
        E.g. if we have productions
            sum: ?Int ==> sum(?Map(Int))
            flatten: ?Map(Any) ==> flatten(?Map(Map(Any)))
            +: ?Int ==> ?Int + ?Int
            var: ?Map(Map(Int)) ==> v
        then
            - Map(Map(Int)) is accessible via production "var"
            - Map(Int) is accessible via production flatten, because 
                Map(Map(Any)) >= Map(Map(Int)), and Map(Map(Int)) is accessible
            - Int is accessible via production "sum", because Map(Int) is accessible
            - Int is accessible also via "+", because Int itself is accessible
        """
        accessible: Dict[Type, Set[Production]] = dict()

        # populate with productions whose left is concrete and right is complete
        for p in self._l:
            if p.left.is_concrete() and p.right().is_complete():
                if p.left not in accessible:
                    accessible[p.left] = set()
                accessible[p.left].add(p)
                # print("{} via {}".format(p.left, p))
            
        while True:
            fixpoint = True
            for p in self._l:
                ts = p.right().holes() # type holes
                tc = [t for t in ts if t.is_concrete()] # concrete holes
                ta = [t for t in ts if not t.is_concrete()] # abstract holes
                
                if len(ta) > 1:
                    raise Exception("Does not support >1 abstract type holes")
                
                # left is accessible if
                # - all concrete type holes are accessible
                # - the only abstract type hole (if any) can be instantiated by some accessible type
                left_c = None
                if all([t in accessible for t in tc]):
                    if len(ta) == 0:
                        assert(p.left.is_concrete())
                        left_c = p.left
                    else:
                        ta_ok = [ta[0].subsume(t) for t in accessible]
                        if any(ta_ok):
                            sigma = [s for s in ta_ok if s][0]
                            left_c = p.left.subst(sigma)
                if left_c:
                    if left_c not in accessible:
                        accessible[left_c] = set()
                    if p not in accessible[left_c]:
                        # print("{} via {}".format(left_c, p))
                        accessible[left_c].add(p)
                        fixpoint = False
            if fixpoint: break

        return accessible

    def __getitem__(self, i):
        if isinstance(i, Type):
            return self._accessible[i]
        elif type(i) == int:
            return self._l[i]
        elif type(i) == str:
            return self._d[i]
        else:
            return dict()[i]

    def __iter__(self):
        return iter(self._l)
    
    def __len__(self):
        return len(self._l)
    
    def print_accessible(self):
        fmt = "{:>20} {:>50}"
        print(fmt.format("<type>", "<productions>"))
        for typ, ps in self._accessible.items():
            print(fmt.format(repr(typ), repr(ps)))
    
    def print_ps(self):
        fmt = "{:>5} {}"
        for i, p in enumerate(self._l):
            print(fmt.format(i, str(p)))
    
    @staticmethod
    def generic() -> List[Production]:
        return [
            Production(
                Bool(),
                lambda: Infix("&&", Bool(), Hole(Bool()), Hole(Bool())),
                "and"),
            Production(
                Bool(),
                lambda: Infix("<=", Bool(), Hole(Int()), Hole(Int())),
                "leq"),
            Production(
                Int(),
                lambda: Infix("+", Int(), Hole(Int()), Hole(Int())),
                "add"),
            Production(
                Map(Any()), 
                lambda: Unary("flatten", Map(Any()), Hole(Map(Map(Any())))), 
                "flatten"),
            Production(
                Int(),
                lambda: Unary("sum", Int(), Hole(Map(Int()))),
                "sum"),
            Production(
                Int(),
                lambda: Leaf("0", Int()),
                "zero"),
        ]