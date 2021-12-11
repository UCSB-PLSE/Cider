
from abc import ABC
class ContractGraph(ABC):
  @abstractmethod
  def __init__(self):
    raise NotImplemented
  
  @abstractmethod
  def vertex_tokens(self):
    raise NotImplemented
  
  @abstractmethod
  def edge_tokens(self):
    raise NotImplemented
  
  @abstractmethod
  def identifier_tokens(self):
    raise NotImplemented
  
  
  def insert_padding_node(self, arg_padding_token="<PAD>"):
    """Inserts an extra node at the very beginning (index=0) with padding token as label"""
    # assertions come first
    for i in range(len(self.ig.vs)):
        assert self.ig.vs[i].index == i, "Indices mismatch for node at {}, expected {} but got {}.".format(i, i, self.ig.vs[i].index)
    for i in range(len(self.ig.es)):
        assert self.ig.es[i].index == i, "Indices mismatch for edge at {}, expected {} but got {}.".format(i, i, self.ig.es[i].index)
    # first export nodes as records
    node_attributes = self.ig.vs["token"]
    edge_attributes = self.ig.es["token"]
    edge_tuples = [ self.ig.es[i].tuple for i in range(len(self.ig.es)) ]
    # inset one extra padding node at the beginning (index=0)
    node_attributes = [arg_padding_token] + node_attributes
    # shift all the indices in edge info by +1 to the id
    edge_tuples = [ 
        (edge_tuples[i][0]+1, edge_tuples[i][1]+1)
        for i in range(len(edge_tuples))
    ]
    new_igraph = ig.Graph(
        directed=True,
        n=len(node_attributes),
        vertex_attrs={"token": node_attributes},
        edges=edge_tuples,
        edge_attrs={"token": edge_attributes},
    )

    # process/shift the remaining components
    new_e2n = { dkey:self.e2n[dkey]+1 for dkey in self.e2n.keys() }
    new_e2r = {
        [ self.e2r[dkey][i]+1 for i in range(len(self.e2r[dkey])) ]
        for dkey in self.e2r.keys()
    }
    new_root_id = self.root_id + 1

    self.ig = new_igraph
    self.e2n = new_e2n
    self.e2r = new_e2r
    self.root_id = new_root_id
  
  
  def add_reversed_edges(self):
    """Inserts reversed edges by exchanging the source and target and marking the edge label as REV_???"""
    # first export nodes as records
    node_attributes = self.ig.vs["token"]
    edge_attributes = self.ig.es["token"]
    edge_tuples = [ self.ig.es[i].tuple for i in range(len(self.ig.es)) ]
    assert len(edge_attributes)==len(edge_tuples)
    # then add reversed edges
    ext_edge_attributes = []
    ext_edge_tuples = []
    for i in range(len(edge_tuples)):
        ext_edge_attributes.append( "REV_{}".format(edge_attributes[i]) )
        ext_edge_tuples.append( (edge_tuples[i][1], edge_tuples[i][0]) )
    
    self.ig = ig.Graph(
        directed=True,
        n=len(node_attributes),
        vertex_attrs={"token": node_attributes},
        edges=edge_tuples+ext_edge_tuples,
        edge_attrs={"token": edge_attributes+ext_edge_attributes},
    )