from collections import defaultdict
from tree import ASTNode
from tree_typed import TASTNode

T_BOOL = "bool"
T_INT = "int"
T_ANY = "any"
T_MAP = lambda res: ("map", res)
is_map = lambda t: type(t) == tuple and len(t) == 2 and t[0] == "map"
is_base = lambda t: t == T_INT or t == T_BOOL
destruct_map = lambda t: t[1]
rule_var = lambda v: "var_" + v

def erase_types(typed_productions):
    return {
        hole: {
            rule: (kind, [c for c,_ in children])
            for ty, ty_prod in prod.items()
            for rule, (kind, children) in ty_prod.items()
        }
        for hole, prod in typed_productions.items()
    }

def expandable(r, hole, t): 
    assert(is_concrete(t))
    try:
        return len(r[hole][t]) > 0
    except KeyError:
        return False

def _subtype(t, t_concrete, post=lambda x: x):
    if t == T_ANY:
        return t_concrete
    elif is_base(t) and t == t_concrete:
        return t_concrete
    elif is_map(t) and is_map(t_concrete):
        return post(_subtype(destruct_map(t), destruct_map(t_concrete), post=post))
    else:
        raise Exception(f"Ill-formed type: {t}")

def subtype(t, t_concrete):
    try:
        return _subtype(t, t_concrete, post=T_MAP)
    except Exception:
        return False

def subst(t, o, n):
    if t == o:
        return n
    elif is_base(t):
        return t
    elif is_map(t):
        return T_MAP(subst(destruct_map(t), o, n))
    else:
        raise Exception(f"Ill-formed type: {t}")

def subtype_get_subst(t, t_concrete):
    return lambda t0: subst(t0, T_ANY, _subtype(t, t_concrete))
    
# list(map(print, [
#     subtype(T_INT, T_BOOL),
#     subtype(T_INT, T_INT),
#     subtype(T_ANY, T_INT),
#     subtype(T_ANY, T_MAP(T_ANY)),
#     subtype(T_MAP(T_MAP(T_ANY)), T_MAP(T_MAP(T_MAP(T_INT)))),
#     subtype(T_MAP(T_BOOL), T_MAP(T_INT))
# ]))
# print()

def is_concrete(t):
    if t == T_ANY:
        return False
    elif is_base(t):
        return True
    elif is_map(t):
        return is_concrete(destruct_map(t))
    else:
        raise Exception(f"No such type: {t}")

# list(map(print, [
#     is_concrete(T_INT),
#     is_concrete(T_BOOL),
#     is_concrete(T_MAP(T_MAP(T_INT))),
#     is_concrete(T_MAP(T_MAP(T_ANY))),
# ]))
# print()

# r = defaultdict(lambda: defaultdict(set))
# r["t"][T_MAP(T_MAP(T_INT))] = {0}
# r["t"][T_MAP(T_BOOL)] = {0}
# r["t"][T_MAP(T_INT)] = {0}

def instantiate(r, t, hole):
    """Instantiate an abstract type `t` at `hole`"""
    assert(not is_concrete(t))
    res = []
    for t2 in r[hole]:
        if len(r[hole][t2]) == 0:
            continue
        assert(is_concrete(t2))
        t1 = subtype(t, t2)
        if t1:
            res.append(t1)
    return res

# list(map(print, [
#     instantiate(T_ANY, "t"),
#     instantiate(T_MAP(T_ANY), "t"),
#     instantiate(T_MAP(T_MAP(T_ANY)), "t"),
# ]))
# print()

def expand_abstract(r, hole, t):
    return [t1 for t1 in instantiate(r, t, hole) if expandable(r, hole, t1)]

def partition(xs, pred):
    A, B = [], []
    for x in xs:
        if pred(x):
            A.append(x)
        else:
            B.append(x)
    return A, B

# list(map(print, [
#     expandable_abstract("t", T_ANY),
#     expandable_abstract("t", T_MAP(T_ANY)),
#     expandable_abstract("t", T_MAP(T_MAP(T_ANY))),
#     expandable_abstract("t", T_MAP(T_MAP(T_MAP(T_ANY)))),
# ]))
# print()

# is_hole = lambda kind: kind in ["var", "e", "t", "phi"]

def node_expandable(r, hole, t, kind, children):
    is_hole = lambda kind: kind in r
    if is_hole(kind):
        hole_2 = kind
        assert(not is_concrete(t))
        return expand_abstract(r, hole_2, t) # expand into another hole with the same type
    else:
        child_concrete, child_abstract = partition(children, lambda c_t: is_concrete(c_t[1]))
        
        # currently only support single occurrence of type variable
        assert(len(child_abstract) <= 1)

        if not all([expandable(r,c,c_t) for c,c_t in child_concrete]):
            return []
        elif len(child_abstract) == 0:
            return [t]
        else:
            return [subtype_get_subst(c_t, c_t_concrete)(t) for c, c_t in child_abstract for c_t_concrete in expand_abstract(r, c, c_t)]

# list(map(print, [
#     node_expandable("t", T_ANY, "sum", [("t", T_MAP(T_INT))]),
#     node_expandable("t", T_MAP(T_ANY), "flatten", [("t", T_MAP(T_MAP(T_ANY)))])
# ]))
# print()

def print_ptp(ptp):
    for hole in ptp:
        for t in ptp[hole]:
            if len(ptp[hole][t]) > 0:
                print(f"{hole} : {t}".ljust(40), end="")
                print(ptp[hole][t])

def to_prediction_table(typed_productions, stovar_types, hole_order=["e", "t", "phi"]):
    r = defaultdict(lambda: defaultdict(set))
    for v,t in stovar_types:
        r["var"][t] = set([rule_var(v)])    
    print_ptp(r)

    is_hole = lambda kind: kind in typed_productions
    i = 0
    # run until fixpoint
    while True:
        print(f"Iteration {i}")
        change = False
        for hole in hole_order:
            for t in typed_productions[hole]:
                for rule, (kind, children) in typed_productions[hole][t].items():
                    if rule in r[hole][t]:
                        continue
                    assert(not is_hole(kind) or children == [])
                    
                    for t1 in node_expandable(r, hole, t, kind, children):
                        print(f"{hole} gets {t1} because of child \"{kind}\"")
                        if rule not in r[hole][t1]:
                            change = True
                            r[hole][t1].add(rule)

        if not change:
            break
        i += 1

    return r


def typed_productions():
    return {
        "phi": {
            T_BOOL: {
                "phi_and": ("&&", [("phi", T_BOOL), ("phi", T_BOOL)]),
                "phi_leq": ("<=", [("t", T_INT), ("t", T_INT)]),
            },
        },
        "t": {
            T_ANY: {
                "t_e": ("e", []),
            },
            T_INT: {
                "t_sum": ("sum", [("t", T_MAP(T_INT))]),
            },
            T_MAP(T_ANY): {
                "t_flatten": ("flatten", [("t", T_MAP(T_MAP(T_ANY)))])
            }
        },
        "e": {
            T_INT: {
                "e_0": ("0", []),
                "e_add": ("+", [("e", T_INT), ("e", T_INT)])
            },
            T_ANY: {
                "e_var": ("var", []),
            }
        },
    }

def to_typed_tree_productions(tp, start, start_typ):
    display_kind = lambda kind, typ, cs: f"{kind}"
    display_hole = lambda kind, typ, cs: f"?{kind}"
    display_unary = lambda op, typ, cs: f"{op}({str(cs[0])})"
    display_infix = lambda infix, typ, cs: f"{str(cs[0])} {infix} {str(cs[1])}"

    holes = set(tp.keys())
    rights = {kind: children \
        for hole in tp \
            for typ in tp[hole] \
                for rule, (kind, children) in tp[hole][typ].items()}
    non_leaves = set([kind for (kind, children) in rights.items() if children != []]) - holes
    leaves = set([kind for (kind, children) in rights.items() if children == []]) - holes
    
    get_hole_node = lambda hole, typ: TASTNode(hole, typ, is_hole=True, fmt=display_hole)
    get_leaf_node = lambda kind, typ: TASTNode(kind, typ, fmt=display_kind)
    get_unary_node = lambda kind, typ, children: TASTNode(kind, typ, children=children, fmt=display_unary)
    get_infix_node = lambda kind, typ, children: TASTNode(kind, typ, children=children, fmt=display_infix)
    nodes = dict()

    res = dict()
    for hole in tp:
        if hole not in res:
            res[hole] = dict()
        for typ in tp[hole]:
            if typ not in res[hole]:
                res[hole][typ] = dict()
            for rule, (kind, children) in tp[hole][typ].items():
                print(f"hole: {hole.ljust(10)} type: {str(typ).ljust(10)} rule: {rule.ljust(10)}", end="")

                if kind in tp: # hole
                    assert(len(children) == 0)
                    hole2 = kind
                    node = lambda hole=hole2, typ=typ: get_hole_node(hole, typ)
                else:
                    if len(children) == 0: # leaf
                        node = lambda kind=kind, typ=typ: get_leaf_node(kind, typ)
                    else:
                        if len(children) > 2:
                            raise NotImplementedError
                        children = [get_hole_node(c, c_t) for (c, c_t) in children]
                        if len(children) == 1:
                            node = lambda kind=kind, typ=typ, children=children: get_unary_node(kind, typ, children)
                        elif len(children) == 2:
                            node = lambda kind=kind, typ=typ, children=children: get_infix_node(kind, typ, children)

                res[hole][typ][rule] = node
                print(node())

    return lambda hole=start, typ=start_typ: get_hole_node(hole, typ), res

to_typed_tree_productions(typed_productions(), "phi", T_BOOL)

def to_tree_productions(productions, start):
    display_typ = lambda typ, cs: typ
    display_hole = lambda typ, cs: f"?{typ}"
    display_unary = lambda op, cs: f"{op}({str(cs[0])})"
    display_infix = lambda infix, cs: f"{str(cs[0])} {infix} {str(cs[1])}"
    
    holes = set(productions.keys())
    rights = {typ: children for (left, rules) in productions.items() for (typ, children) in rules.values()}
    non_leaves = set([typ for (typ, children) in rights.items() if children != []]) - holes
    leaves = set([typ for (typ, children) in rights.items() if children == []]) - holes
    nodes = dict()
    
    # print(f"Holes: {', '.join(holes)}")
    for h in holes:
        nodes[h] = lambda h=h: ASTNode(h, is_hole=True, fmt=display_hole)
        # print("\n".join(map(str, [(n, nodes[n]()) for n in holes if n in nodes])))
        # print()

    # print(f"Non-leaves: {', '.join(non_leaves)}")
    
    for n in non_leaves:
        children = [nodes[h]() for h in rights[n]]
        if n == "<=":
            print()
        if len(children) == 1:
            fmt = display_unary
        elif len(children) == 2:
            fmt = display_infix
        else:
            # print("Non-leaf with >= 3 children not supported")
            assert(False)
        nodes[n] = lambda n=n, children=children, fmt=fmt: ASTNode(n, children=children, fmt=fmt)
    
        # print("\n".join(map(str, [(n, nodes[n]()) for n in non_leaves if n in nodes])))
        # print()
    
    # print(f"Leaves: {', '.join(leaves)}")
    for l in leaves:
        nodes[l] = lambda l=l: ASTNode(l, fmt=display_typ)
    
    start_node = nodes[start]
    
    tree_productions = {
        hole: {
            rule: nodes[typ] for rule, (typ, children) in rules.items()
        } for hole, rules in productions.items()
    }
    # for hole, rules in tree_productions.items():
    #     for rule, node in rules.items():
    #         n = node()
    #         n_str = str(n)
    #         print(hole, rule, n_str)
    return start_node, tree_productions

def insert_vars(productions, vars):
    productions["var"] = {f"{rule_var(v)}": (v, []) for v in vars}

def insert_typed_vars(typed_productions, var_type_lst):
    assert("var" not in typed_productions)
    typed_productions["var"] = dict()
    var_productions = typed_productions["var"]
    for v, t in var_type_lst:
        if t not in var_productions:
            var_productions[t] = dict()
        var_productions[t][rule_var(v)] = (v, [])

def test():
    print()

if __name__ == "__main__":
    # test()
    tp = typed_productions()
    stovars = {
        "bal": T_MAP(T_MAP(T_INT)),
        "tot": T_INT
    }
    insert_typed_vars(tp, stovars)

    ptp = to_prediction_table(tp, stovars)
    print_ptp(ptp)
    for hole, rules in to_tree_productions(erase_types(tp), start="phi")[1].items():
        for rule, node in rules.items():
            print(hole.ljust(10), rule.ljust(10), node())
