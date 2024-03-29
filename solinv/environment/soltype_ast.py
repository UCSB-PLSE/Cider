from copy import deepcopy
import igraph as ig

soltype_vertex_tokens = [
            '<CONTRACT>',
            # expressions
            '<VAR>',
            'CInt', 'CBool', 'CArrZero',
            "Unary_not", "!=", ">=", "<=", ">", "<", "==", "||", "&&", 
            "+", "-",  "*", "/", "%", "**",
            "+=", "-=", "*=", "/=",
            'EMapInd', 'EField', 'EHavoc',
            # lvalues
            'LvInd', 'LvFld',
            # statements
            'SAsn', 'SCall', 'SDecl', 'SIf', 'SReturn', 'SWhile', 'SHavoc', 'SAbort',
            # declarations
            'DCtor',
            'DFun', 'DFun_arg',
            'DStruct',
            # types
            'TyMapping', 'TyStruct', 'TyArray', 'TyAddress', 'TyInt', 'TyInt32', 'TyInt64', 'TyInt120', 'TyInt256', 'TyBool', 'TyByte']

soltype_edge_tokens = [
            # expressions
            'Var_name', 'Unary_e', 'Binary_lhs', 'Binary_rhs', 
            'EField_fld', 'EField_struct', 'EMapInd_ind', 'EMapInd_map', 
            # lvalues
            'LvFld_fld', 'LvFld_struct', 'LvInd_ind', 'LvInd_map', 
            # statements
            'SAsn_lhs', 'SAsn_rhs', 'SCall_args', 'SCall_args_first', 'SCall_args_next', 'SCall_name', 
            'SIf_cond', 'SIf_else', 'SIf_then', 'SIf_then_first', 'SIf_then_next', 'SIf_else_first', 'SIf_else_next',
            'SWhile_cond', 'SWhile_body_first', 'SWhile_body_next',
            # declarations
            'DCtor_args', 'DCtor_body', 
            'DFun_args', 'DFun_body', 'DFun_name',
            'DFun_args_first', 'DFun_args_next', 'DFun_arg_name', 'DFun_arg_type',
            'DFun_body_first', 'DFun_body_next', 
            'DVar_expr', 'DVar_name', 'DVar_type', 
            # types
            'TyMapping_dst', 'TyArray_elem',
            # contract
            'DFun_first', 'DFun_next',
            # special
            'contents']

def add_reversed_edges(arg_igraph):
    """Inserts reversed edges by exchanging the source and target and marking the edge label as REV_???"""
    # first export nodes as records
    node_attributes = arg_igraph.vs["token"]
    edge_attributes = arg_igraph.es["token"]
    edge_tuples = [ arg_igraph.es[i].tuple for i in range(len(arg_igraph.es)) ]
    assert len(edge_attributes)==len(edge_tuples)
    # then add reversed edges
    ext_edge_attributes = []
    ext_edge_tuples = []
    for i in range(len(edge_tuples)):
        ext_edge_attributes.append( "REV_{}".format(edge_attributes[i]) )
        ext_edge_tuples.append( (edge_tuples[i][1], edge_tuples[i][0]) )
    return ig.Graph(
        directed=True,
        n=len(node_attributes),
        vertex_attrs={"token": node_attributes},
        edges=edge_tuples+ext_edge_tuples,
        edge_attrs={"token": edge_attributes+ext_edge_attributes},
    )

def insert_padding_node(arg_igraph, arg_var_to_vertex, arg_padding_token="<PAD>"):
    """Inserts an extra node at the very beginning (index=0) with padding token as label"""
    # assertions come first
    for i in range(len(arg_igraph.vs)):
        assert arg_igraph.vs[i].index == i, "Indices mismatch for node at {}, expected {} but got {}.".format(i, i, arg_igraph.vs[i].index)
    for i in range(len(arg_igraph.es)):
        assert arg_igraph.es[i].index == i, "Indices mismatch for edge at {}, expected {} but got {}.".format(i, i, arg_igraph.es[i].index)
    # first export nodes as records
    node_attributes = arg_igraph.vs["token"]
    edge_attributes = arg_igraph.es["token"]
    edge_tuples = [ arg_igraph.es[i].tuple for i in range(len(arg_igraph.es)) ]
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
    new_var_to_vertex = { dkey:arg_var_to_vertex[dkey]+1 for dkey in arg_var_to_vertex.keys() }
    # new_e2r = {
    #     [ arg_e2r[dkey][i]+1 for i in range(len(arg_e2r[dkey])) ]
    #     for dkey in arg_e2r.keys()
    # }
    return new_igraph, new_var_to_vertex

def extract_tokens(d, f):
    """Extract tokens from an JSON AST, using extraction function f"""

    def inner(d):
        if type(d) not in [dict, list]:
            return set()
        elif type(d) == list:
            res = [inner(x) for x in d]
            return set().union(*res)
        else:
            res = [inner(v) for v in d.values()]
            return set(f(d)).union(*res)

    return inner(d)

def linearize(ast):
    """Turn lists into linked lists in an AST"""
    if type(ast) != dict:
        return ast
    worklist = [(k, v) for k, v in ast.items() if type(v) == list and len(v) > 0]
    for k, v in worklist:
        subtrees = list(map(linearize, ast[k]))
        # Make new tokens for head and next pointers
        # header pointer token = key to list + _first
        # next pointer token = key to list + _next
        # Note: In the future, we might want to change them to _first and _next (without the key prefix)
        ast[k + "_first"] = subtrees[0]
        try:
            for i in range(len(subtrees) - 1):
                subtrees[i][k + "_next"] = subtrees[i + 1]
        except TypeError:
            print(subtrees)
    # clean up
    for k, _ in worklist:
        del ast[k]
    return ast

def make_contract_ast(l_dfun):
    """Make an contract from a list of function declarations"""
    return {"tag": "<CONTRACT>", "DFun": l_dfun}

def preprocess_DFun_args(dfun):
    """Preprocess function arguments in function declaration `dfun`"""
    assert "DFun_args" in dfun
    dfun["DFun_args"] = [
        {"tag": "DFun_arg", "DFun_arg_name": name, "DFun_arg_type": t}
        for name, t in dfun["DFun_args"]
    ]
    return dfun

def label_vertices(ast, vi, vertices, var_v):
    """Label each node in the AST with a unique vertex id
    vi : vertex id counter
    vertices : list of all vertices (modified in place)
    """

    def inner(ast):
        nonlocal vi
        if type(ast) != dict:
            if type(ast) == list:
                # print(vi)
                pass
            return ast
        ast["vertex_id"] = vi
        vertices.append(ast["tag"])
        # if not (ast['tag'] in ['EVar', 'LvVar'] and ast['contents'] in var_v):
        vi += 1
        for k, v in ast.items():
            if k != "tag":
                inner(v)
        return ast

    return inner(ast)

def label_edges(ast, ei, edges, var_v):
    """Label each edge in the AST with a unique edge id
    ei : edge id counter
    edges : list of all edges (modified in place)
    """

    def inner(ast, p=None, edge_token=None):
        nonlocal ei
        if type(ast) != dict:
            if type(ast) == list:
                # print(ei)
                pass
            return ast
        # if this is a storage variable, connect to it directly
        if ast["tag"] == "<VAR>" and ast["Var_name"] in var_v:
            vi = var_v[ast["Var_name"]]
        else:
            vi = ast["vertex_id"]
        if p is not None:
            edges.append(((p, vi), edge_token))
            ei += 1
        # recurse
        for k, v in ast.items():
            if k != "tag":
                inner(v, vi, k)
        return ast

    return inner(ast)

def get_soltype_graph(contract_json):
    contract_name, contents = contract_json
    find = lambda l, tag: [x for x in l if type(x) == dict and x["tag"] == tag]
    # constructors
    l_dctor = find(contents, "DCtor")[0]
    # storage variables
    l_dvar = find(contents, "DVar")
    # functions
    l_dfun = find(contents, "DFun")

    # preprocess function arguments
    l_dfun2 = [preprocess_DFun_args(dfun) for dfun in l_dfun]
    contract = make_contract_ast(l_dfun2)

    # turn lists into linked lists
    contract2 = linearize(contract)

    # sanity check: vertex tokens and edge tokens
    v_tokens = sorted(
        list(
            extract_tokens(contract2, lambda d: [v for k, v in d.items() if k == "tag"])
        )
    )
    e_tokens = sorted(
        list(
            extract_tokens(contract2, lambda d: [k for k, v in d.items() if k != "tag"])
        )
    )
    # TODO: assert v_tokens \subset self.reserved_vertex_token_list
    # TODO: assert e_tokens \subset self.reserved_edge_token_list

    def inverse(d):
        """Return the inverse dictionary of d"""
        return {v: k for k, v in d.items()}

    # vertex index |-> storage variable name
    v_var = {i: v["DVar_name"] for i, v in enumerate(l_dvar)}
    # storage variable name |-> vertex index
    var_v = inverse(v_var)

    # reserve the first several vertices for storage variables
    vertices = sorted(["<VAR>" for _ in var_v])
    # print(vertices)
    # populate the vertex list
    contract3 = label_vertices(deepcopy(contract2), len(vertices), vertices, var_v)

    # populate the edge list
    edges = list()
    contract4 = label_edges(deepcopy(contract3), 0, edges, var_v)

    contract_ast, vs, es = contract4, vertices, edges
    g = ig.Graph(
        directed=True,
        n=len(vs),
        vertex_attrs={"token": vs},
        edges=[e for e, _ in es],
        edge_attrs={"token": [tk for _, tk in es]},
    )

    g.delete_vertices([v for v in g.vs if v.degree() == 0])

    return var_v, g
