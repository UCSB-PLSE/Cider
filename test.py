import argparse
import numpy as np
import os
import random

# ray related utils
import ray
from ray import tune
from ray.rllib.agents import ppo, dqn
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print, UnifiedLogger

# trinity related utils
import solinv.tyrell.spec as S
import solinv.tyrell.dsl as D
from solinv.tyrell.spec import sort
from solinv.tyrell.interpreter import InvariantInterpreter
from solinv.environment import InvariantEnvironment
from solinv.model import InvariantTGN, InvariantGCN, TestNN

def test(contract, version):
    spec = S.parse_file("./dsls/abstract0.tyrell")
    start_type = spec.get_type("Expr", sort.BOOL)
    interpreter = InvariantInterpreter()
    env_config = {
        "spec": spec,
        "start_type": start_type,
        "max_step": 6,
        # version options are: 0.4.26, 0.5.17, 0.6.12
        "contracts": [
          (f, v) for f,v in
          [
          (contract, ver)
        ]],
        "num_tests": 0,
        "interpreter": interpreter
    }
    # need to construct the vocab first to provide parameters for nn
    tmp_environment = InvariantEnvironment(config=env_config)

    # ray.init(local_mode=True)
    # use non local mode to enable GPU
    ray.init()
    # ModelCatalog.register_custom_model("invariant_tgn", InvariantTGN)
    # ModelCatalog.register_custom_model("invariant_gcn", InvariantGCN)
    ModelCatalog.register_custom_model("test_nn", TestNN)

    rl_config = ppo.DEFAULT_CONFIG.copy()
    rl_config = {
        "env": InvariantEnvironment,
        "env_config": env_config,
        "model": {
            "custom_model": "test_nn",
            "custom_model_config": {
                "num_token_embeddings": len(tmp_environment.token_list),
                "token_embedding_dim": 16,
                "token_embedding_padding_idx": tmp_environment.token_dict["<PAD>"],
                "conv_heads": 4,
                "conv_dropout": 0,
                # "conv_heads": 8,
                # "conv_dropout": 0.6,

                "invariant_hidden_dim": 64,
                "invariant_out_dim": 32,

                "action_out_dim": len(tmp_environment.action_list),

                # this provides a shared access to the contract related utils
                "environment": InvariantEnvironment,
            },
        },
        "num_workers": 1,
        "num_gpus": args.ngpu,
        "framework": "torch",
    }

    trainer = ppo.PPOTrainer(
        env=InvariantEnvironment, 
        config=rl_config,
    )
    trainer.train()
    ray.shutdown()