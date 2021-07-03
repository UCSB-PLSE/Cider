import graph

class ASTNode:
    def __init__(self, kind, children=[], is_hole=False, fmt=None):
        self.kind = kind
        self.is_hole = is_hole
        self.children = children
        self.fmt = fmt

    def can_expand(self, left):
        return left in self.collect()

    def expand(self, left, right):
        def f(node):
            if node.kind == left:
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
            return self.kind
        else:
            return "{}({})".format(self.kind, ", ".join([str(c) for c in self.children]))

    
    def __str__(self):
        if self.fmt is None:
            return repr(self)
        else:
            return self.fmt(self.kind, [str(c) for c in self.children])
    
    def collect(self):
        acc = set()
        def f(node):
            acc.add(node.kind)
            for child in node.children:
                f(child)
        f(self)
        return acc
    
    def is_complete(self):
        return not self.is_hole and all((child.is_complete() for child in self.children))
    
    def to_graph(self):
        special_nodes = {v: "special__" + v for v in self.collect()}
        g = graph.MyGraph()
        def f(node):
            g.add_undirected(node.kind, special_nodes[node.kind])
            for child in node.children:
                g.add_undirected(node.kind, child.kind)
                f(child)
        f(self)
        return g

if __name__ == '__main__':
    print("tree.py")