from typ import Bool, Int, Map, Any
from tree import Hole, Node
from production import *
from productions import *

p_lst = Productions.generic()
p_lst.extend([
    Production(Map(Map(Int())), lambda: Leaf("usr", Map(Map(Int()))), "usr"),
    Production(Map(Int()), lambda: Leaf("balances", Map(Int())), "balances"),
    Production(Int(), lambda: Leaf("total", Int()), "total")
])

ps = Productions(p_lst)
ps.print_ps()

seq = ["leq", "add", "zero", "total", "sum", "balances"]

node: Node = Hole(Bool())
for s in seq:
    print(ps[s])
    node = node.expand(ps[s])
    print(node)

v = ps.vocabulary()
