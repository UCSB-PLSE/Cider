from collections import defaultdict, Counter

class MyGraph:
    def __init__(self):
        self.V = set()
        self.E = defaultdict(set)
    
    def add_directed(self, src, dst):
        self.V.add(src)
        self.V.add(dst)
        self.E[src].add(dst)
    
    def add_undirected(self, u, v):
        self.add_directed(u, v)
        self.add_directed(v, u)
    
    def __str__(self):
        res = "V: {}\n".format(self.V)
        for s in self.E:
            for d in self.E[s]:
                res += "{} --> {}\n".format(s, d)
        return res
    
    def to_adj_matrix(self):
        labels = list(self.V)
        labels.sort()
        label = {i: l for i,l in enumerate(labels)}
        V = list(range(len(labels)))
        return [[int(label[v] in self.E[label[u]]) for v in V] for u in V], label
    
    def weisfeiler_lehman(self, n=20, k=10):
        V_sorted = sorted(list(self.V))
        C = [dict() for _ in range(20+1)]
        C[0] = {v: 1 for v in self.V}
        color_count = 2
        code_to_color = dict()
        change = False
        res = defaultdict(set)
        res[1] = set(self.V)
        for i in range(1,n+1):
            for v in sorted(list(self.V)):
                C_old, C_curr = C[i-1], C[i]
                l = C_old[v]
                m = tuple(sorted([C_old[u] for u in self.E[v]]))
                c = (l, m) # new compressed code
                if c in code_to_color:
                    C_curr[v] = code_to_color[c]
                    res[code_to_color[c]].add(v)
                else:
                    code_to_color[c] = color_count
                    C_curr[v] = color_count
                    res[color_count].add(v)
                    color_count += 1
                    change = True
                # print("{}: old = {}, multiset = {}, new = {}".format(v, l, m, C_curr[v]))
            if not change:
                break
        res = [len(res[c]) for c in sorted(list(res))][:k]
        res += [0] * (k - len(res))
        return res

if __name__ == "__main__":
    import numpy as np
    sim = lambda u,v: 1 - np.dot(u,v)/sum(u)/sum(v)

    g = MyGraph()
    g.add_undirected('a', 'b')
    g.add_undirected('a', 'c')
    g.add_undirected('b', 'd')
    print(g)
    g1_embedding = g.weisfeiler_lehman()
    print(g1_embedding, len(g1_embedding))
    g1_adj, g1_label = g.to_adj_matrix()

    g.add_undirected('a', 'a')
    g2_embedding = g.weisfeiler_lehman()
    print(g2_embedding, len(g2_embedding))
    print(sim(g1_embedding, g2_embedding))
    # g2_adj, g2_label = g.to_adj_matrix()

    # # from grakel import Graph
    # # from grakel.kernels import ShortestPath
    # # G1 = Graph(initialization_object=g1_adj, node_labels=g1_label)
    # # G2 = Graph(initialization_object=g2_adj, node_labels=g2_label)
    # # sp_kernel = ShortestPath(normalize=True)
    # # print(sp_kernel.fit_transform([G1]))
    # # print(sp_kernel.transform([G2]))

