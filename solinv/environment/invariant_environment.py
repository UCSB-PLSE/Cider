import copy
import pickle
import subprocess
import random
import torch
import numpy as np
from collections import defaultdict
from typing import List, Any, Union, Dict

import gym
import igraph
from gym.utils import seeding

from ..tyrell import spec as S
from ..tyrell.spec import sort
from ..tyrell import dsl as D
from ..tyrell.interpreter import InvariantInterpreter
from ..tyrell.dsl import Node, HoleNode
from ..tyrell.dsl.utils import derive_dfs, get_hole_dfs

from .invariant_heuristic import InvariantHeuristic
from .error import EnvironmentError

from .soltype_ast import get_soltype_graph, insert_padding_node, add_reversed_edges, soltype_edge_tokens, soltype_vertex_tokens

from ..tyrell.spec import sort
from .svgraph import StoVarGraph
from .solid import Solid
from .contract import Contract

def nonzero(xs):
    return [i for i, x in enumerate(xs) if x != 0]

class InvariantEnvironment(gym.Env):
    # note: class static variable
    #       used to track previously sampled sequences, for coverage based exploration
    sampled_action_seqs = {}
    cached_contract_utils = {}

    CONTRACT_MAX_IDS   = 100
    CONTRACT_MAX_NODES = 1000

    def __init__(self, config: Dict[str, Any], is_test=False):
        self.config = config
        self.tspec = config["spec"]
        self.builder = D.Builder(self.tspec)
        self.start_type = config["start_type"]
        self.max_step = config["max_step"]
        self.interpreter = config["interpreter"]
        self.is_test = is_test
        self.solid = Solid()

        # ================== #
        # vocabulary related #
        # ================== #
        # solType slim AST tokens
        self.special_token_list = ["<PAD>", "<ID>", "<REF>"]
        self.reserved_identifier_token_list = [] # TODO: populate this

        self.reserved_vertex_token_list = sorted(StoVarGraph.VERTEX_TOKENS)
        self.reserved_edge_token_list = sorted(StoVarGraph.EDGE_TOKENS)
        # self.reserved_vertex_token_list = sorted(soltype_vertex_tokens)
        # self.reserved_edge_token_list = sorted(soltype_edge_tokens)
        
        # extend the edge token with reversed version
        tmp_reversed_edge_token_list = ["REV_{}".format(p) for p in self.reserved_edge_token_list]
        self.reserved_edge_token_list = sorted( self.reserved_edge_token_list+tmp_reversed_edge_token_list )
            
        # every token in the token list should have a fixed embedding for
        self.base_token_list = self.special_token_list \
                             + self.reserved_identifier_token_list \
                             + self.reserved_vertex_token_list \
                             + self.reserved_edge_token_list
        fixed_list = [x for x in self.tspec.productions() if not(x.is_enum() and "<VAR" in x._get_rhs())]
        self.token_list = self.base_token_list + fixed_list
        self.token_dict = {self.token_list[i]:i for i in range(len(self.token_list))}
    
        # fixme: here we setup all contracts first to prevent contract id not found error in non local mode of RLlib
        for i, c in enumerate(config["contracts"]):
            print("Initializing contract", c[0], c[1])
            self.setup(config, arg_id=i)


        # this caches contract utils for faster switching
        # between contracts in training between different rollouts
        self.curr_contract_id = None # need to reset
        _ = self.reset()

        # inherited variables
        # note: for action space, we are using the maximum available productions
        #       in practice, we may be just using a subset of them, i.e., with some of the flex action not used
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.observation_space = gym.spaces.Dict({
            "start": gym.spaces.Box(0,1,shape=(1,),dtype=np.int32),
            "contract_id": gym.spaces.Box(0, InvariantEnvironment.CONTRACT_MAX_IDS, shape=(1,), dtype=np.int32),
            "action_mask": gym.spaces.Box(0, 1, shape=(len(self.action_list),), dtype=np.int32), # for output layer, no need to + len(sptok_list)
            
            # action_seq series: returns the current action sequence, in two channels, with <PAD> for padding and values belonging to other channel
            "action_seq@token_channel": gym.spaces.Box(0, len(self.token_list), shape=(self.max_step,), dtype=np.int32),
            "action_seq@node_channel": gym.spaces.Box(0, InvariantEnvironment.CONTRACT_MAX_NODES, shape=(self.max_step,), dtype=np.int32),

            # all_actions series: for dynamic action output, remain the same for the same contract, same channelling style
            "all_actions@token_channel": gym.spaces.Box(0, len(self.token_list), shape=(len(self.action_list),), dtype=np.int32),
            "all_actions@node_channel": gym.spaces.Box(0, InvariantEnvironment.CONTRACT_MAX_NODES, shape=(len(self.action_list),), dtype=np.int32),
        })
    
    def toggle_test(self):
        self.is_test = True

    def setup(self, arg_config, arg_id=None):
        if arg_id is None:
            num_contracts = len(arg_config["contracts"])
            num_tests = arg_config["num_tests"]
            num_train = num_contracts - num_tests
            if self.is_test:
                self.curr_contract_id = random.choice(list(range(num_train, num_contracts)))
            else:
                # if no contract is specified, randomly choose one
                self.curr_contract_id = random.choice(list(range(num_train)))

        else:
            self.curr_contract_id = arg_id

        if self.curr_contract_id in InvariantEnvironment.cached_contract_utils.keys():
            # directly pull from cache
            cached = InvariantEnvironment.cached_contract_utils[self.curr_contract_id]
            
            # spec related
            self.private_tspec = cached["private_tspec"]
            self.action_list = cached["action_list"]
            self.action_dict = cached["action_dict"]
            self.fixed_action_list = cached["fixed_action_list"]
            self.fixed_action_dict = cached["fixed_action_dict"]
            self.flex_action_list = cached["flex_action_list"]
            self.flex_action_dict = cached["flex_action_dict"]
            
            self.contract = cached["contract"]
            self.var_to_vertex = cached["var_to_vertex"]
            self.contract_observed = cached["contract_observed"]
            self.stovar_list = cached["stovar_list"]
            self.stovar_dict = cached["stovar_dict"]
            self.flex_action_to_stovar = cached["flex_action_to_stovar"]
            self.stovar_to_flex_action = cached["stovar_to_flex_action"]
            self.contract_baseline_scores = cached["contract_baseline_scores"]
            self.action_masks = cached["action_masks"]
            self.shadow_actions = cached["shadow_actions"]
            self.inv_cache = cached["inv_cache"]
            self.svg = cached["svg"]
        else:
            # need to start a new process
            self.private_tspec = copy.deepcopy(self.tspec)
            # action list that contains every production rule in the dsl
            self.action_list = self.private_tspec.productions()
            self.action_dict = {self.action_list[i]:i for i in range(len(self.action_list))}
            # a fixed action is shared across different benchmarks
            self.fixed_action_list = list(filter(
                lambda x: not(x.is_enum() and "<VAR" in x._get_rhs()),
                self.action_list,
            ))
            # hack
            # self.fixed_action_list[-2].set_lhs_sort(sort.BOOL) # second to last fixed action is EnumType -> true
            self.fixed_action_list[-1].set_lhs_sort(sort.INT) # last  fixed action is EnumType -> 0
            
            self.fixed_action_dict = {self.fixed_action_list[i]:i for i in range(len(self.fixed_action_list))}
            # a flex action is bounded with a stovar dynamically for different benchmarks
            self.flex_action_list = list(filter(
                lambda x: x not in self.fixed_action_list,
                self.action_list,
            ))
            self.flex_action_dict = {self.flex_action_list[i]:i for i in range(len(self.flex_action_list))}
            # note: re-order action list to have fixed+flex order
            self.action_list = self.fixed_action_list + self.flex_action_list
            self.action_dict = {self.action_list[i]:i for i in range(len(self.action_list))}
            # note: see notes in `observe_action_seq`
            assert self.action_list[0].name == "empty", "Need `empty` as the first production rule in DSL."
            contract_path, solc_version, svg_str = arg_config["contracts"][self.curr_contract_id]
            # contract_path, solc_version, *svg_str = arg_config["contracts"][self.curr_contract_id]
            self.contract = Contract(contract_path, solc_version)

            # ================ #
            # contract related #
            # ================ #
            
            self.svg = StoVarGraph(self.contract, svg_str)
            # tokenize the target contract
            self.solid.use_solc_version(self.contract.version)
            self.contract_json = self.solid.contract_json(self.contract)
            
            # var_to_vertex: variable name -> node id
            #      e.g., {'_balances': 0, '_totalSupply': 4, 'account': 5, 'value': 6}
            # - an variable name is NOT always a stovar

            # self.var_to_vertex, ig = get_soltype_graph(self.contract_json)
            self.var_to_vertex, ig = self.svg.get_igraph(self.contract_json)
            # note: add reversed edges
            ig = add_reversed_edges(ig)
            # note: adding an extra padding node
            ig, self.var_to_vertex = insert_padding_node(ig, self.var_to_vertex)

            # self.contract_networkx = igraph.Graph.to_networkx(ig)
            # map tokens to corresponding ids (no variable will show up since the graph is already anonymous)
            self.contract_encoded_igraph = ig.copy()
            for p in self.contract_encoded_igraph.vs:
                try:
                    p["token"] = self.token_dict[p["token"]]
                except KeyError:
                    raise NotImplementedError("Unsupported token, got: {}.".format(p["token"]))
            for p in self.contract_encoded_igraph.es:
                p["token"] = self.token_dict[p["token"]]
            self.contract_observed = {
                "x": torch.tensor(self.contract_encoded_igraph.vs["token"]).long(),
                "edge_attr": torch.tensor(self.contract_encoded_igraph.es["token"]).long(),
                # shape is: (2, num_edges)
                "edge_index": torch.tensor([
                    [self.contract_encoded_igraph.es[i].source for i in range(len(self.contract_encoded_igraph.es))],
                    [self.contract_encoded_igraph.es[i].target for i in range(len(self.contract_encoded_igraph.es))],
                ]).long()
            }

            # get stovars list
            self.stovar_list, self.stovar_sorts = self.solid.storage_variables(self.contract)
            self.stovar_dict = {self.stovar_list[i]:i for i in range(len(self.stovar_list))}
            # check for enough var production rules in the dsl
            enum_expr = self.private_tspec.get_type("EnumExpr", sort.ANY)
            for i in range(len(self.stovar_list)):
                _ = self.private_tspec.get_enum_production_or_raise(enum_expr, "<VAR{}>".format(i))
            # update types of used vars
            for i in range(len(self.stovar_list)):
                s = self.stovar_sorts[i]
                p = self.private_tspec.get_enum_production_or_raise(enum_expr, "<VAR{}>".format(i))
                p.set_lhs_sort(s)
            # change unused EnumExpr's sort from ANY to BOTTOM
            for p in self.private_tspec.productions():
                if isinstance(p.lhs, S.type.EnumType) and p.lhs.sort == sort.ANY:
                    p.set_lhs_sort(sort.BOTTOM)
            # establish the flex-stovar bindings
            self.flex_action_to_stovar = {
                self.private_tspec.get_enum_production_or_raise(enum_expr, "<VAR{}>".format(i)) : self.stovar_list[i]
                for i in range(len(self.stovar_list))
            }
            self.stovar_to_flex_action = { self.flex_action_to_stovar[dkey]:dkey for dkey in self.flex_action_to_stovar.keys() }


            self.action_masks = self.compute_action_masks()
            self.shadow_actions = self.get_shadow_actions()
            self.inv_cache = dict()

            # generate the baseline score (i.e., score for the "true" invariant)
            # a tuple of ( hard, tot_hard, soft, tot_soft )
            self.contract_baseline_scores = self.check(self.contract, "true", arg_silent_mode=True)
            # note: do you need to assert the inferiority of the scores of baseline?

            print("# ======")
            print("# contract: {}\n# var_to_vertex: {}\n# num_nodes: {}\n# num_edges: {}\n# baseline scores: {}".format(
                self.contract.path, self.var_to_vertex,
                len(ig.vs), len(ig.es),
                self.contract_baseline_scores
            ))
            print("# start type:", self.start_type)
            print("# stovars:", ", ".join(self.stovar_list))
            print("# action masks:")
            # for t, mask in self.action_masks.items():
            #     print(t, nonzero(mask))
            self.print_action_masks(self.action_masks)
            print("# ======")

            # store to cache
            InvariantEnvironment.cached_contract_utils[self.curr_contract_id] = {}
            cached = InvariantEnvironment.cached_contract_utils[self.curr_contract_id]
            # spec related
            cached["private_tspec"] = self.private_tspec
            cached["action_list"] = self.action_list
            cached["action_dict"] = self.action_dict
            cached["fixed_action_list"] = self.fixed_action_list
            cached["fixed_action_dict"] = self.fixed_action_dict
            cached["flex_action_list"] = self.flex_action_list
            cached["flex_action_dict"] = self.flex_action_dict

            cached["contract"] = self.contract
            cached["contract_observed"] = self.contract_observed
            cached["var_to_vertex"] = self.var_to_vertex
            cached["stovar_list"] = self.stovar_list
            cached["stovar_dict"] = self.stovar_dict
            cached["flex_action_to_stovar"] = self.flex_action_to_stovar
            cached["stovar_to_flex_action"] = self.stovar_to_flex_action
            cached["contract_baseline_scores"] = self.contract_baseline_scores
            cached["action_masks"] = self.action_masks
            cached["shadow_actions"] = self.shadow_actions
            cached["inv_cache"] = self.inv_cache
            cached["svg"] = self.svg
            # self.print_cached_action_masks()

        # ====== #
        # basics #
        # ====== #
        # initialize internal variables
        self.curr_trinity_inv = None # invariant in trinity node structure
        # action_seq: represented using ids from action_list, for internal tracking of the environment
        self.curr_action_seq = None

    def observe_action_seq(self, arg_action_seq):
        # turn designated nn_seq into its observed form (channelling)
        # returns: two channels of the same action sequence

        # note: <PAD> will usually here point to 0, which in action list should be `empty` production rule
        #       we also abuse a bit here that the following three tokens all have "padding" functionalities with id 0:
        #       - `empty` production rule
        #       - <PAD> in token
        #       - a pddding node with <PAD> label 
        #         (a padding node should always have <PAD> label, but a node with <PAD> is not always a padding node)
        ret_seq_token, ret_seq_node = [], []
        for p in arg_action_seq:
            if p >= len(self.fixed_action_list):
                # flex action
                if self.action_list[p] in self.flex_action_to_stovar.keys():
                    ret_seq_token.append(self.token_dict["<PAD>"])
                    ret_seq_node.append(self.var_to_vertex[self.flex_action_to_stovar[self.action_list[p]]])
                else:
                    # this action does not have corresponding stovar (the prod is not useful), use padding instead
                    ret_seq_token.append(self.token_dict["<PAD>"])
                    ret_seq_node.append(self.token_dict["<PAD>"])
            else:
                # fixed action
                ret_seq_token.append(p)
                ret_seq_node.append(self.token_dict["<PAD>"])
        return ret_seq_token, ret_seq_node

    def pad_to_length(self, arg_obj, arg_length):
        return arg_obj + [self.token_dict["<PAD>"] for _ in range(arg_length-len(arg_obj))]

    def trinity_inv_to_debugging_inv(self, arg_trinity_inv):
        # debugging inv still retains the recursive trinity structure
        # this will replace all replacable <VAR?> with binded stovar
        tmp_inv = str(arg_trinity_inv)
        for prod in self.flex_action_to_stovar.keys():
            tmp_inv = tmp_inv.replace(prod._get_rhs(), self.flex_action_to_stovar[prod])
        return tmp_inv

    def trinity_inv_to_verifier_inv(self, arg_trinity_inv):
        # verifier inv will be the string that is directly provided to the verifier
        tmp_inv0 = self.interpreter.eval(arg_trinity_inv)
        tmp_inv1 = self.trinity_inv_to_debugging_inv(tmp_inv0) # compatible reuse
        return tmp_inv1

    def record_action_seq(self, arg_action_seq):
        # add the sequence to the class level recorder and count
        tup_seq = tuple(arg_action_seq)
        if self.curr_contract_id not in InvariantEnvironment.sampled_action_seqs.keys():
            InvariantEnvironment.sampled_action_seqs[self.curr_contract_id] = {}
        if tup_seq not in InvariantEnvironment.sampled_action_seqs[self.curr_contract_id].keys():
            InvariantEnvironment.sampled_action_seqs[self.curr_contract_id][tup_seq] = 0
        InvariantEnvironment.sampled_action_seqs[self.curr_contract_id][tup_seq] += 1
    
    def compute_action_masks(self):
        # print(f"compute_action_masks for contract {self.curr_contract_id}")
        paths = defaultdict(set)
        ps = []
        for p in self.action_list:
            if p.lhs.is_enum():
                if p.lhs.sort != sort.BOTTOM:
                    paths[p.lhs].add(p)
            else:
                ps.append(p)

        def print_paths(paths):
            print('# ===')
            for t in paths:
                print(t)
                for p in paths[t]:
                    print(p)
                    print('# ---')
            print('# ===')
    
        # for p in ps:
            # print(p.lhs, "->", ", ".join(map(str, p.rhs)))

        while True:
            change = False
            for p in ps:
                sigmas = []
                for ta in p.rhs:
                    for tc in paths:
                        if tc <= ta and len(paths[tc]) > 0 and p not in paths[tc]:
                            sigmas.append(ta.subsume(tc))
                if len(sigmas) == 0: continue
                for sigma in sigmas:
                    rhs_concrete = [ta.subst(sigma) for ta in p.rhs]
                    lhs_concrete = p.lhs.subst(sigma)
                    # print(lhs_concrete, "->", ", ".join(map(str, rhs_concrete)))
                    # print()
                    if all([len(paths[t]) > 0 for t in rhs_concrete]) and p not in paths[lhs_concrete]:
                        paths[lhs_concrete].add(p)
                        change = True
            if not change:
                break
        
        # print_paths(paths)
        action_list = self.fixed_action_list + self.flex_action_list
        masks = defaultdict(lambda: [0 for _ in range(len(action_list))])
        for t in paths:
            mask = [int(p in paths[t]) for p in action_list]
            masks[t] = mask
            # print(t, nonzero(mask))
        return masks

    def print_action_masks(self, masks):
        for t, mask in masks.items():
            valid_actions = []
            for i in range(len(mask)):
                if mask[i] == 0: continue
                if isinstance(self.action_list[i], S.production.FunctionProduction):
                    a = self.action_list[i].name
                elif i >= len(self.fixed_action_list):
                    a = self.stovar_list[i-len(self.fixed_action_list)]
                # elif i == len(self.fixed_action_list) - 1:
                #     a = "true"
                else:
                    a = "0"
                valid_actions.append(a)
            print("{:<30} {}".format(str(t), ", ".join(valid_actions)))
    
    def get_action_mask(self, arg_type):
        # get action mask that allows for a specific type
        return self.action_masks[arg_type]

    def print_cached_action_masks(self):
        for k in self.cached_contract_utils:
            print("Cached action masks for contract", k)
            for t, mask in self.cached_contract_utils[k]["action_masks"].items():
                print(t, nonzero(mask))
            print()
    
    def get_shadow_actions(self):
        shadow_ps = dict()
        sorts = set()
        for p in self.action_list:
            if p.lhs.sort.is_concrete():
                sorts.add(p.lhs.sort)
                sorts |= set([r.sort for r in p.rhs if isinstance(r, S.type.Type)])
        for p in self.action_list:
            if not p.lhs.sort.is_concrete():
                if p not in shadow_ps:
                    shadow_ps[p] = dict()
                for s in sorts:
                    if not s <= p.lhs.sort: continue
                    p0 = copy.deepcopy(p)
                    e0 = copy.deepcopy(p.lhs)
                    e0._sort = s
                    sigma = p0.lhs.subsume(e0)
                    p0._lhs = p0.lhs.subst(sigma)
                    p0._rhs = [t.subst(sigma) if isinstance(t, S.type.Type) else t for t in p0.rhs]
                    shadow_ps[p][s] = p0
        return shadow_ps


    def is_max(self):
        '''
        Returns whether or not the length of action sequence has already reached the preset limit.
        '''
        # -1 since we always have a <SOS> at the beginning
        return len(self.curr_action_seq) >= self.max_step

    def is_done(self):
        '''
        Returns whether or not there's still any remaining hole in the current invariant.
        '''
        next_hole = get_hole_dfs(self.curr_trinity_inv)
        if next_hole is None:
            return True
        else:
            return False
    
    def reset(self):
        # note: this should return data structure as defined by self.observation_space
        #       not only the state (but also including any action mask)
        self.setup(self.config)
        self.curr_trinity_inv = HoleNode(type=self.start_type)
        self.curr_action_seq = []
        tmp_action_seq_token, tmp_action_seq_node = self.observe_action_seq(self.curr_action_seq)
        tmp_all_actions_token, tmp_all_actions_node = self.observe_action_seq(list(range(len(self.action_list))))
        return {
            "start": [1],
            "contract_id": [self.curr_contract_id],
            "action_mask": self.get_action_mask(self.start_type),
            "action_seq@token_channel": self.pad_to_length(tmp_action_seq_token, self.max_step),
            "action_seq@node_channel": self.pad_to_length(tmp_action_seq_node, self.max_step),
            "all_actions@token_channel": self.pad_to_length(tmp_all_actions_token, self.max_step),
            "all_actions@node_channel": self.pad_to_length(tmp_all_actions_node, self.max_step),
        }

    def check(self, arg_contract: Contract, arg_verifier_inv: str, arg_silent_mode: bool=False):
        if arg_verifier_inv in self.inv_cache:
            result = self.inv_cache[arg_verifier_inv]
        else:
            result = self.solid.check(arg_contract, arg_verifier_inv)
        hard_ok, hard, soft_ok, soft = result
        if not arg_silent_mode:
            # print()
            print("# [debug][verifier-result] ----------> hard: {:d}/{:d} ({:+d}), soft: {:d}/{:d} ({:+d}) <----------".format(
                hard_ok, hard, hard_ok - self.contract_baseline_scores[0],
                soft_ok, soft, soft_ok - self.contract_baseline_scores[2],
            ))
        if hard_ok + soft_ok == hard + soft:
            # input("Found the ground truth!")
            pass
        self.inv_cache[arg_verifier_inv] = result
        return list(result)

    # the action id here is for the action_list / action space for sure    
    def step(self, arg_action_id: int):
        '''
        returns: [state, reward, terminate, info]
        '''
        if arg_action_id >= len(self.action_list):
            raise EnvironmentError("Action id is not in range, required: [0, {}), got: {}".format(len(self.action_list), arg_action_id))
        
        if self.is_done():
            raise EnvironmentError("the invariant is already complete; no action is required.")

        
        # perform the action: derive method will raise exceptions by itself
        # try:
        p = self.action_list[arg_action_id]
        if not p.lhs.sort.is_concrete():
            hole = get_hole_dfs(self.curr_trinity_inv)
            p = self.shadow_actions[p][hole.type.sort]
        # try:
        # print("# [debug] contract: {}, curr_inv: {}, action: {}".format(self.curr_contract_id, self.curr_trinity_inv, p))
        sts, new_inv = derive_dfs(self.builder, self.curr_trinity_inv, p)
        # print("# [debug] contract: {}, new_inv: {}".format(self.curr_contract_id, new_inv))
        # except:
        #     # Exception: Types don't match, expect Empty, got Expr
        #     self.curr_action_seq = self.curr_action_seq + [arg_action_id]
        #     print("# [debug][done/exception] contract: {}, seq: {}, inv(before): {}".format(
        #         self.curr_contract_id, self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
        #     ))
        #     tmp_action_seq_token, tmp_action_seq_node = self.observe_action_seq(self.curr_action_seq)
        #     tmp_all_actions_token, tmp_all_actions_node = self.observe_action_seq(list(range(len(self.action_list))))
        #     return [
        #         {
        #             # you can't fill any hole since the seq terminates with an exception
        #             "start": [1],
        #             "contract_id": [self.curr_contract_id],
        #             "action_mask": [0 for _ in range(len(self.action_list))], 
        #             "action_seq@token_channel": self.pad_to_length(tmp_action_seq_token, self.max_step),
        #             "action_seq@node_channel": self.pad_to_length(tmp_action_seq_node, self.max_step),
        #             "all_actions@token_channel": self.pad_to_length(tmp_all_actions_token, self.max_step),
        #             "all_actions@node_channel": self.pad_to_length(tmp_all_actions_node, self.max_step),
        #         }, 
        #         0.0, # reward 
        #         True, # terminate
        #         {}, # info
        #     ]

        if not sts:
            raise EnvironmentError("node is not expanded, check the implementation")
        # if you are here, then the derivation is successful
        # then refresh state
        self.curr_trinity_inv = new_inv
        self.curr_action_seq = self.curr_action_seq + [arg_action_id]
        
        # ================================ #
        # ====== reward computation ====== #
        # ================================ #
        # hm: heuristic multiplier (default 1.0, any heuristic failing will make it 0.1)
        # rm: repeat multiplier (default 1.0, computed by 1.0/<times>)
        # all rewards will be multiplied by hm and rm
        # there are different cases
        # if the invariant is complete
        #   - if it fails some heuristics: 1.0
        #   - else
        #     - if it fails the checking: 0.1
        #     - if it passes the checking: 10.0 * percentage_of_constraints_passed 
        # if the invariant is not complete
        #   - but it reaches the max allowed step: 0.0 (which means it should've completed before)
        #   - and it still can make more steps: 0.1 (continue then)
        tmp_done = self.is_done()
        tmp_max = self.is_max()
        tmp_action_mask = None # TBD later
        tmp_terminate = None # TBD later
        tmp_reward = None # TBD later
        tmp_heuristic_multiplier = 1.0 # helper for reward shaping of partial heuristics
        tmp_repeat_multiplier = 1.0 # helper for reward shaping of coverage-based exploration
        heuristic_list = [
            # InvariantHeuristic.no_enum2expr_root(self.curr_trinity_inv),
            InvariantHeuristic.no_duplicate_children(self.curr_trinity_inv)
        ]
        if not all(heuristic_list):
            tmp_heuristic_multiplier = 0.1
        # satisfy all partial heuristics
        if tmp_done:
            self.record_action_seq(self.curr_action_seq)
            tmp_repeat_multiplier = 1.0/InvariantEnvironment.sampled_action_seqs[self.curr_contract_id][tuple(self.curr_action_seq)]
            # done, should check the heuristics first
            if not all(heuristic_list):
                # some heuristics won't fit, prevent this invariant from going to checker
                print("# [debug][heuristic][hm={}][rm={:.2f}] contract: {}, seq: {}, inv(before): {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_contract_id,
                    self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
                ))
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                tmp_reward = 1.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
            else:
                # all good, go to the checker
                print("# [debug][done][hm={}][rm={:.2f}] contract: {}, seq: {}, inv(before): {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_contract_id,
                    self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
                ))
                tmp_verifier_inv = self.trinity_inv_to_verifier_inv(self.curr_trinity_inv)
                tmp_reslist = self.check(self.contract, tmp_verifier_inv)
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                if tmp_reslist is None:
                    tmp_reward = 1.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
                else:
                    if tmp_reslist[0]+tmp_reslist[2]==tmp_reslist[1]+tmp_reslist[3]:
                        # completely correct, remove rm
                        tmp_reward = 10.0*(tmp_reslist[0]+tmp_reslist[2])/(tmp_reslist[1]+tmp_reslist[3]) * tmp_heuristic_multiplier
                    elif tmp_reslist[0]==tmp_reslist[1]:
                        # hard constraints are all satisfied, further categorize
                        if tmp_reslist[2] > self.contract_baseline_scores[2]:
                            # soft score better than baseline score, remove rm
                            tmp_reward = 10.0*(tmp_reslist[0]+tmp_reslist[2])/(tmp_reslist[1]+tmp_reslist[3]) * tmp_heuristic_multiplier
                        else:
                            # soft score is no better than baseline score, need rm to get rid of this
                            tmp_reward = 10.0*(tmp_reslist[0]+tmp_reslist[2])/(tmp_reslist[1]+tmp_reslist[3]) * tmp_heuristic_multiplier * tmp_repeat_multiplier
                    else:
                        # some hard constraints are not satisfied, still need rm
                        tmp_reward = 10.0*(tmp_reslist[0]+tmp_reslist[2])/(tmp_reslist[1]+tmp_reslist[3]) * tmp_heuristic_multiplier * tmp_repeat_multiplier
        else:
            if self.is_max():
                self.record_action_seq(self.curr_action_seq)
                tmp_repeat_multiplier = 1.0/InvariantEnvironment.sampled_action_seqs[self.curr_contract_id][tuple(self.curr_action_seq)]
                print("# [debug][max][hm={}][rm={:.2f}] contract: {}, seq: {}, inv: {}".format(
                    tmp_heuristic_multiplier, tmp_repeat_multiplier, 
                    self.curr_contract_id,
                    self.curr_action_seq, self.trinity_inv_to_debugging_inv(self.curr_trinity_inv),
                ))
                tmp_action_mask = [0 for _ in range(len(self.action_list))]
                tmp_terminate = True
                tmp_reward = 0.0 * tmp_heuristic_multiplier * tmp_repeat_multiplier
            else:
                # tmp_node here must not be None since it's not done yet
                tmp_node = get_hole_dfs(self.curr_trinity_inv)
                tmp_action_mask = self.get_action_mask(tmp_node.type)
                tmp_terminate = False
                tmp_reward = 0.1 * tmp_heuristic_multiplier * tmp_repeat_multiplier

        tmp_action_seq_token, tmp_action_seq_node = self.observe_action_seq(self.curr_action_seq)
        tmp_all_actions_token, tmp_all_actions_node = self.observe_action_seq(list(range(len(self.action_list))))
        return [
            {
                "start": [1],
                "contract_id": [self.curr_contract_id],
                "action_mask": tmp_action_mask, 
                "action_seq@token_channel": self.pad_to_length(tmp_action_seq_token, self.max_step),
                "action_seq@node_channel": self.pad_to_length(tmp_action_seq_node, self.max_step),
                "all_actions@token_channel": self.pad_to_length(tmp_all_actions_token, self.max_step),
                "all_actions@node_channel": self.pad_to_length(tmp_all_actions_node, self.max_step),
            }, 
            tmp_reward, 
            tmp_terminate, 
            {}, # info
        ]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        pass

    def close(self):
        pass