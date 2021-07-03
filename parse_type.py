from lark import Lark
import typ as T

type_grammar = """
    start: t
    
    t : "address" -> int
      | "uint256" -> int
      | "mapping(" t "=>" t ")" -> mapping

    %import common.WS
    %ignore WS
"""

parser = Lark(type_grammar)

def parse(type_str):
    parse_tree = parser.parse(type_str)
    def f(t):
        if t.data == "int":
            return T.Int()
        elif t.data == "mapping":
            return T.Map(f(t.children[1]))
    return f(parse_tree.children[0])


def test(): 
    print(parse("mapping(address => mapping(address => uint256))"))
    print(parse("mapping(uint256 => uint256)"))

if __name__ == '__main__':
    test()
    # main()