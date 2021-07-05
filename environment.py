import re
import subprocess
from typing import List
import numpy as np

import graph
import parse_type
from production import Production
from productions import Productions
from tree import Leaf, Hole, Node
import typ as T

def get_vars(filename) -> List[Production]:
    """Get the storage variables"""
    res = subprocess.run(f"liquidsol-exe {filename} --task stovars", shell=True, capture_output=True)
    assert(res.returncode == 0)
    res = res.stdout.decode("utf-8")
    res = res.rstrip().split("\n")[1:]
    assert(len(res) % 2 == 0)
    p_lst = []
    n = len(res) // 2
    for i in range(n):
        var = res[i]
        typ = parse_type.parse(res[n + i])
        if typ:
            p = Production(typ, lambda var=var, typ=typ: Leaf(var, typ), var)
            p_lst.append(p)
    return p_lst


def check(filename, inv):
    """Check if the invariant holds"""
    ret = subprocess.run(f"liquidsol-exe {filename} --task check --check-inv '{inv}'", shell=True, capture_output=True)
    # print(ret)
    if ret.returncode != 0:
        return None
    ret = ret.stdout.decode("utf-8")
    hard_ok, hard = re.search("Hard: ([0-9]+) / ([0-9]+)", ret).groups()
    soft_ok, soft = re.search("Soft: ([0-9]+) / ([0-9]+)", ret).groups()
    return list(map(int, [hard_ok, hard, soft_ok, soft]))


class Environment:
    def __init__(self, filename, state_size=10, maximum_expansions=10, start_type=T.Bool()):
        self.filename = filename
        self.maximum_expansions = maximum_expansions
        
        self.start_node = lambda: Hole(start_type)
        self.p_lst = Productions.generic()
        self.p_lst.extend(get_vars(self.filename))
        self.ps = Productions(self.p_lst)
    
        self._state_size = state_size
        _, _, self.soft_baseline, self.soft = check(filename, "true")
        print("Soft baseline: {}/{}".format(self.soft_baseline, self.soft))
        
        self.reset()
    
    def reset(self):
        self.done = False
        self.ast: Node = self.start_node()
        self.encode()
        self.expand_count = 0
        self.ps.print_accessible()
        self.ps.print_ps()
        return self.state
    
    def encode(self):
        self.state = self.ast.to_graph().weisfeiler_lehman(k=self._state_size)
    
    @property
    def action_size(self):
        return len(self.ps)
    
    @property
    def state_size(self):
        return self._state_size
    
    def get_mask(self):
        holes = self.ast.holes()
        legal = set().union(*[self.ps.access(hole) for hole in holes])
        mask = np.array([p in legal for _,p in self.ps])
        return mask
    
    def step(self, action_i):
        assert(not self.done)
        action_i = int(action_i)
        assert(action_i < len(self.ps))

        mask = self.get_mask()

        if not mask[action_i]:
            print("Invalid production:", self.ps.nth(action_i)[1].name)
            next_state = self.state
            reward = -1
            done = True
        else:
            _,p = self.ps.nth(action_i)
            ast = self.ast._expand(p)
            # assert(ast)
            if ast is None:
                next_state = self.state
                reward = -1
                done = True
            else:
            
                # update ast and encoding
                self.ast = ast
                self.encode()
                self.expand_count += 1

                print(self.ast)

                if not self.ast.is_complete() and self.expand_count < self.maximum_expansions:
                    next_state = self.state
                    reward = -0.01
                    done = False
                
                else:
                    # complete ast
                    next_state = self.state
                    done = True
                    check_res = check(self.filename, str(self.ast))
                    if check_res is None or self.expand_count >= self.maximum_expansions: # syntax or semantic errors
                        print("Too many expansions:", self.ast)
                        reward = -1
                    else:
                        hard_ok, hard, soft_ok, soft = check_res
                        assert(hard > 0)
                        if hard_ok < hard: # hard constraint violated
                            reward = -1
                        else:
                            reward = 0
                            soft_ok = max(0, soft_ok - self.soft_baseline)
                            soft = max(0, soft - self.soft_baseline)

                            if soft > 0:
                                reward += soft_ok / soft
                                if soft_ok == soft:
                                    print("YAYYYYY!", self.ast)
        if done:
            self.done = True
        return next_state, reward, done, None


if __name__ == '__main__':
    f = "../SolidTypes/test/regression/good/mint_MI.sol"
    res = check(f, "true")
    print(res)
    
    env = Environment(f)

    steps = [
        1,4,6,7
    ]
    print("ast:", env.ast)
    for s in steps:
        print(env.step(s)[1:])
        print(env.ast)
        print()