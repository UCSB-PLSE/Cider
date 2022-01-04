import igraph as ig
import subprocess
from .contract import Contract
from .solid import Solid

class StoVarGraph:
  TOKEN_LOCAL = "<LOCAL>"
  TOKEN_VAR = "<VAR>"
  TOKEN_FUNC = "<FUNC>"
  VERTEX_TOKENS = [TOKEN_LOCAL, TOKEN_VAR]
  EDGE_TOKENS = [TOKEN_FUNC]

  def node(self, name):
    return [v for i,v in enumerate(self.g.vs) if v['name'] == name][0]
  def ind(self, name):
    return [i for i,v in enumerate(self.g.vs) if v['name'] == name][0]

  def __init__(self, solid: Solid, contract: Contract, flow_edges_str):
    self.sto_vars = solid.storage_variables(contract)
    self.flow_edges, flow_vars = self.parse_flow_edges(flow_edges_str)
    self.vars = list(self.sto_vars | flow_vars)
    # print(self.vars)
    self.g = ig.Graph(directed=True)
    self.g.add_vertices(len(self.vars))
    for i, v in enumerate(self.g.vs):
      n = self.vars[i]
      v['name'] = n
      if StoVarGraph.is_local(n):
        self.node(n)['token'] = StoVarGraph.TOKEN_LOCAL
      else:
        assert n in self.sto_vars, f"{n} is not a storage variable"
        self.node(n)['token'] = StoVarGraph.TOKEN_VAR


    def add(n1, n2, fn):
      """Add an edge between the node named n1 and the node named n2 due to function fn"""
      self.g.add_edge(self.ind(n1), self.ind(n2))
      
      if fn is not None:
        e = [e for e in self.g.es if e.tuple == (self.ind(n1), self.ind(n2))][0]
        e["name"] = fn
        e["token"] = StoVarGraph.TOKEN_FUNC
    
    for n1, n2, fn in self.flow_edges:
      add(n1, n2, fn)
  
  @staticmethod
  def is_local(n):
    return "$" in n

  def parse_flow_edges(self, flow_edges_str):
    fn = None
    flow_edges = []
    flow_vars = set()
    for i, l in enumerate(filter(lambda s: s != '', flow_edges_str.split('\n'))):
      l = l.strip()
      if l == '': continue
      if l[-1] == ":":
        fn = l.rstrip(":")
      else:
        src, dst = l.split(", ")
        src = fn + src if StoVarGraph.is_local(src) else src
        dst = fn + dst if StoVarGraph.is_local(dst) else dst
        flow_edges.append((src, dst, fn))
        flow_vars.add(src)
        flow_vars.add(dst)
    return flow_edges, flow_vars

  def get_igraph(self, contract_json):
    var_to_vertex = {v['name']: i for i, v in enumerate(self.g.vs)}
    return var_to_vertex, self.g

