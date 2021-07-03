from typ import *
from tree import *
from production import *
from productions import *

p_lst = Productions.generic()
p_lst.append(Production(
    Map(Map(Int())), 
    lambda: Leaf("usr", Map(Map(Int()))), 
    "usr"))

print("\n".join(map(str, p_lst)))

ps = Productions(p_lst)

