import re
import subprocess
import graph
import productions as p
import parse_type

def get_vars(filename):
    """Get the storage variables"""
    res = subprocess.run(f"liquidsol-exe {filename} --task stovars", shell=True, capture_output=True)
    assert(res.returncode == 0)
    res = res.stdout.decode("utf-8")
    res = res.rstrip().split("\n")[1:]
    assert(len(res) % 2 == 0)
    n = len(res) // 2
    name_type_lst = [(res[i], parse_type.parse(res[n + i])) for i in range(n)]
    print(name_type_lst)
    return name_type_lst


def check(filename, inv):
    """Check if the invariant holds"""
    ret = subprocess.run(f"liquidsol-exe {filename} --task check --check-inv '{inv}'", shell=True, capture_output=True)
    print(ret)
    if ret.returncode != 0:
        return None
    ret = ret.stdout.decode("utf-8")
    hard_ok, hard = re.search("Hard: ([0-9]+) / ([0-9]+)", ret).groups()
    soft_ok, soft = re.search("Soft: ([0-9]+) / ([0-9]+)", ret).groups()
    return list(map(int, [hard_ok, hard, soft_ok, soft]))

def pretty(d, indent=0, post=lambda x: x, pre="  "):
    for key, value in d.items():
        print(pre * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1, post, pre)
        else:
            print(pre * (indent+1) + str(post(value)))

class Environment:
    def __init__(self, filename, state_size=10, maximum_expansions=50, start_symbol="phi", start_type=p.T_BOOL):
        self.filename = filename
        self.state_size = state_size
        self.maximum_expansions = maximum_expansions
        self.start_symbol = start_symbol
        self.start_type = start_type
        

        self.typed_productions = p.typed_productions()
        self.var_type_lst = get_vars(self.filename)
        p.insert_typed_vars(self.typed_productions, self.var_type_lst)

        
        self.prediction_table = p.to_prediction_table(self.typed_productions, self.var_type_lst)
        self.start_node, self.tp = p.to_typed_tree_productions(self.typed_productions, start_symbol, start_type)
        
        # fmt = "{hole:>10} {type:>20} {rule:>20} {right:>40}"
        # print(fmt.format(hole="hole", type="type", rule="rule", right="right"))
        # for hole in self.tp:
        #     for typ in self.tp[hole]:
        #         for rule, right in self.tp[hole][typ].items():
        #             print(fmt.format(hole=hole, type=str(typ), rule=rule, right=str(right())))
        # pretty(self.prediction_table)

        self.actions = [((hole, typ), rule) \
            for hole in self.tp \
                for typ in self.tp[hole] \
                    for rule in self.tp[hole][typ]]
        self.rule_index = {self.actions[i][1]: i for i in range(len(self.actions))}
        self.action_size = len(self.actions)
        self.reset()
    
    def reset(self):
        self.ast = self.start_node()
        # print(self.start_node())
        # print("\n".join(["{} {}".format(i, rule) for i, (_, rule) in enumerate(self.actions)]))
        self.encode()
        self.expand_count = 0
        return self.state
    
    def encode(self):
        self.state = self.ast.to_graph().weisfeiler_lehman(k=self.state_size)
    
    def step(self, action_i):
        non_terminals = self.ast.collect()
        valid_rules = set().union(*[self.prediction_table[hole][typ] for hole, typ in non_terminals if hole in self.prediction_table and typ in self.prediction_table[hole]])
        # print(", ".join(valid_rules))
        mask = [self.actions[i][1] in valid_rules for i in range(self.action_size)]

        if not mask[action_i]:
            print("Invalid rule:", self.actions[action_i][1])
            next_state = self.state
            reward = -1
            done = True
        else:
            (hole, typ), rule_name = self.actions[action_i]
            right = self.tp[hole][typ][rule_name]()
            ast, success = self.ast.expand(hole, typ, right)
            assert(success)
            self.ast = ast
            # update ast and encoding
            self.encode()
            self.expand_count += 1

            if not self.ast.is_complete() and self.expand_count < self.maximum_expansions:
                next_state = self.state
                reward = 0.01
                done = False
            
            else:
                # complete ast
                next_state = self.state
                done = True
                check_res = check(self.filename, str(self.ast))
                if check_res is None or self.expand_count >= self.maximum_expansions: # syntax or semantic errors
                    print("Unexpected error:", self.ast)
                    reward = -1
                else:
                    hard_ok, hard, soft_ok, soft = check_res
                    assert(hard > 0)
                    if hard_ok < hard: # hard constraint violated
                        reward = -0.5
                    else:
                        reward = 0
                        if soft > 0:
                            reward += soft_ok / soft
                        else:
                            reward += 1
        return next_state, reward, done, None


if __name__ == '__main__':
    f = "../SolidTypes/test/regression/good/mint_MI.sol"
    res = check(f, "true")
    print(res)
    
    env = Environment(f)

    steps = [
        1, # t <= t
        3, # sum(t) <= t
        2, # sum(e) <= t
        7, # sum(var) <= t
        8, # sum(_balances) <= t
        2, # sum(_balances) <= e
        7, # sum(_balances) <= var
        8, # sum(_balances) <= _totalSupply
    ]
    print("ast:", env.ast)
    for s in steps:
        print(env.step(s)[1:])
        print(env.ast)
        print()