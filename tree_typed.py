import graph
import productions as p

class TASTNode:
    def __init__(self, kind, typ, children=[], is_hole=False, fmt=None):
        self.kind = kind
        self.typ = typ
        self.is_hole = is_hole
        self.children = children
        self.fmt = fmt

    def can_expand(self, left, typ):
        return (left, typ) in self.collect()
    
    def subst_type(self, sigma):
        self.typ = sigma(self.typ)
        for c in self.children:
            c.subst_type(sigma)

    def expand(self, left, typ, right):
        def f(node):
            if node.kind == left and p.subtype(typ, node.typ):
                sigma = p.subtype_get_subst(typ, node.typ)
                right.subst_type(sigma)
                return (right, True)
            for i, child in enumerate(node.children):
                child_1, expanded = f(child)
                if expanded:
                    node.children[i] = child_1
                    return (node, True)
            return (None, False)
        
        self_1, expanded = f(self)
        # if not expanded:
            # print("ERROR: Failed to expand with rule ?{} -> {}".format(left, right))
        return self_1, expanded


    def __repr__(self):
        if self.children == []:
            return f"{self.kind}<{self.typ}>"
        else:
            return "{}<{}>[{}]".format(self.kind, self.typ, ", ".join([str(c) for c in self.children]))

    
    def __str__(self):
        if self.fmt is None:
            return repr(self)
        else:
            return self.fmt(self.kind, self.typ, [str(c) for c in self.children])
    
    def collect(self):
        acc = set()
        def f(node):
            acc.add((node.kind, node.typ))
            for child in node.children:
                f(child)
        f(self)
        return acc
    
    def is_complete(self):
        return not self.is_hole and all((child.is_complete() for child in self.children))
    
    def to_graph(self):
        special_nodes = {v: "special__" + v for v, v_typ in self.collect()}
        g = graph.MyGraph()
        def f(node):
            g.add_undirected(node.kind, special_nodes[node.kind])
            for child in node.children:
                g.add_undirected(node.kind, child.kind)
                f(child)
        f(self)
        return g

if __name__ == '__main__':
    import productions as p
    ma = p.T_MAP(p.T_ANY)
    mma = p.T_MAP(ma)
    flatten = TASTNode("flatten", ma, [TASTNode("t", mma, is_hole=True)])
    left = "t"
    on =  TASTNode("t", p.T_BOOL, is_hole=True)
    print(on.expand(left, ma, flatten))