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
    name_type = [(res[i], parse_type.parse(res[n + i])) for i in range(n)]
    print(name_type)
    return [name for name, t in name_type]


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


class Environment:
    def __init__(self, filename, state_size=10, maximum_expansions=50):
        self.filename = filename
        self.state_size = state_size
        self.maximum_expansions = maximum_expansions
        self.productions = p.generic_productions()
        self.vars = get_vars(self.filename)
        p.insert_vars(self.productions, self.vars)
        self.reset()
    
    def reset(self):
        self.start_node, self.tree_productions = p.to_tree_productions(self.productions, "phi")

        self.actions = [(left, rule_name) for left, rules in self.tree_productions.items() for rule_name in rules]
        self.action_size = len(self.actions)
        self.ast = self.start_node
        self.encode()
        self.expand_count = 0
        return self.state
    
    def encode(self):
        self.state = self.ast.to_graph().weisfeiler_lehman(k=self.state_size)
    
    def step(self, action_i):

        left, rule_name = self.actions[action_i]
        right = self.tree_productions[left][rule_name]()

        ast, success = self.ast.expand(left, right)
        self.ast = ast
        if not success:
            next_state = self.state
            reward = -1
            done = True

        else:
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
        
        # print(self.ast)
        # print(self.state)
        return next_state, reward, done, None


if __name__ == '__main__':
    f = "../SolidTypes/test/regression/good/mint_MI.sol"
    res = check(f, "true")
    print(res)
    env = Environment(f)
    print("\n".join(list(map(str, enumerate(env.actions)))))
    print()

    steps = [0, 2, 1, 6, 8, 1, 6, 9]
    print("ast:", env.ast)
    for s in steps:
        print(env.step(s)[1:])
        print(env.ast)
        print()