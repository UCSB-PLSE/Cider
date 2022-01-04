import subprocess
import json
import igraph as ig

class SVGraph:
  TOKEN_LOCAL = "<LOCAL>"
  TOKEN_SV = "<SV>" # storage var
  TOKEN_EDGE_FLOW = "<FLOW>"
  TOKEN_EDGE_FLOW_INC = "<FLOW_INC>"
  TOKEN_EDGE_FLOW_DEC = "<FLOW_DEC>"
  VERTEX_TOKENS = [TOKEN_LOCAL, TOKEN_SV]
  EDGE_TOKENS = [TOKEN_EDGE_FLOW, TOKEN_EDGE_FLOW_INC, TOKEN_EDGE_FLOW_DEC]

  kind_to_token = {'Flow': TOKEN_EDGE_FLOW, 'FlowInc': TOKEN_EDGE_FLOW_INC, 'FlowDec': TOKEN_EDGE_FLOW_DEC}
  
  def __init__(self, solid, contract):
    flow = solid.flow(contract)
    sv, _ = solid.storage_variables(contract)
    sv_set = set(sv)
    var_set = sv_set | set([v for fn in flow for edge in fn['edges'] for v in [edge['src'], edge['dst']]])
    local_set = var_set - sv_set
    # sort variables first by local/non-local, then by alphabet
    var_list = sorted(sorted(list(var_set)), key=lambda v: v not in sv_set)
    var_dict = {v: i for i, v in enumerate(var_list)}
    g = ig.Graph(directed=True)
    g.add_vertices(len(var_list))
    for i, v in enumerate(g.vs):
      var = var_list[i]
      v['name'] = var
      if var in sv_set:
        v['token'] = SVGraph.TOKEN_SV
      else:
        v['token'] = SVGraph.TOKEN_LOCAL
    for fn in flow:
      sv_used = sv_set & set([v for edge in fn['edges'] for v in [edge['src'], edge['dst']]])
      if len(sv_used) < 2: continue
      for edge in fn['edges']:
        kind = edge['kind']
        src = edge['src']
        dst = edge['dst']
        edge_token = SVGraph.kind_to_token[kind]
        src_i, dst_i = var_dict[src], var_dict[dst]
        if (src_i, dst_i) not in g.es:
          e = g.add_edge(src_i, dst_i)
          e['token'] = edge_token
          fn_name = fn['name'] or '<ctor>'
          e['name'] = fn_name
    self.g = g
    self.var_to_v = {v: i for v, i in var_dict.items() if v in sv_set}
  
  def get_igraph(self, contract_json):
    return self.var_to_v, self.g