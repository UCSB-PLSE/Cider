import argparse
import numpy as np
import os
import random
import argparse
import tempfile

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

# customized logger
# ref: https://stackoverflow.com/questions/62241261/change-logdir-of-ray-rllib-training-instead-of-ray-results
def custom_log_creator(custom_path, custom_str):
    # timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    # logdir_prefix = "{}_{}".format(custom_str, timestr)
    logdir_prefix = "{}_".format(custom_str)
    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)
    return logger_creator


# Train using storage-variable-based graphs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", default=0, type=int, help="how many gpus are there for use, default: 0")
    ap.add_argument("--expname", default="temp", type=str, help="the experiment name, default: temp")
    args = ap.parse_args()

    spec = S.parse_file("./dsls/abstract0.tyrell")
    start_type = spec.get_type("Expr", sort.BOOL)
    interpreter = InvariantInterpreter()
    env_config = {
        "spec": spec,
        "start_type": start_type,
        "max_step": 6,
        "contracts": [
          # ("benchmarks/example1/0x1ccaa0f2a7210d76e1fdec740d5f323e2e1b1672.sol",
          # "0.4.26",
          # """
          # transferFrom:
          #   $_value, allowed
          #   $_value, balances
          # burn:
          #   $_value, totalSupply_
          #   $_value, balances
          # ctor:
          #   $_const, balances
          #   $_const, totalSupply_
          # """),
          # ("benchmarks/example1/0x27f706edde3aD952EF647Dd67E24e38CD0803DD6.sol",
          # "0.4.26",
          # """
          # transferFrom:
          #   $_value, allowed
          #   $_value, balances
          # payable:
          #   totalBonusTokensIssued, $tokensIssued
          #   totalContribution, $tokensIssued
          #   $tokensIssued, totalSupply
          #   $tokensIssued, balances
          # """),
          # ("benchmarks/example1/0x286BDA1413a2Df81731D4930ce2F862a35A609fE.sol",
          # "0.4.26",
          # """
          # transferFrom:
          #   $_value, allowed
          #   $_value, balances
          # mintInternal:
          #   $_amount, totalSupply
          #   $_amount, balances
          # """),
          # ("benchmarks/example1/0x28b5e12cce51f15594b0b91d5b5adaa70f684a02.sol",
          # "0.4.26",
          # """
          # transferFrom:
          #   $_value, allowed
          #   $_value, balances
          # updateWhitelisted:
          #   balances, $prev_amount
          #   $prev_amount, totalSupply_
          #   $tokenAmount, totalSupply_
          # ctor:
          #   INITIAL_SUPPLY, balances
          #   INITIAL_SUPPLY, totalSupply_
          #   """),
          # ("benchmarks/example1/0x3833dda0aeb6947b98ce454d89366cba8cc55528.sol",
          # "0.4.26",
          # """
          # transferFrom:
          #   $_value, allowed
          #   $_value, balances
          # ctor:
          #   decimals, initialSupply
          #   initialSupply, balances
          #   initialSupply, totalSupply
          # """),
          # ("benchmarks/example1/0x4618519de4c304f3444ffa7f812dddc2971cc688.sol",
          # "0.4.26",
          # """
          # transferFrom:
          #   $_value, allowed
          #   $_value, balances
          # ctor:
          #   INITIAL_SUPPLY, balances
          #   INITIAL_SUPPLY, totalSupply_
          # """),
          ("benchmarks/example1/0x539efe69bcdd21a83efd9122571a64cc25e0282b.sol",
          "0.4.26",
          """
          transferFrom:
            $_value, allowed
            $_value, balances
          ctor:
            SUPPLY_CAP, balances
            SUPPLY_CAP, totalSupply
          """),
          # ("benchmarks/example1/0x56D1aE30c97288DA4B58BC39F026091778e4E316.sol",
          # "0.4.26",
          # """
          # transferFrom:
          #   $_value, allowed
          #   $_value, balances
          # mint:
          #   $_amount, balances
          #   $_amount, totalSupply_
          # """),
          # ("benchmarks/example1/0x5ab793e36070f0fac928ea15826b0c1bc5365119.sol",
          # "0.4.26",
          # """
          # transferFrom:
          #   $_value, allowance
          #   $_value, balanceOf
          # mint:
          #   $_amount, balanceOf
          #   $_amount, totalSupply
          # burn:
          #   $_amount, balanceOf
          #   $_amount, totalSupply
          # """),
        ],
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

    # tune.run("PPO", stop={"episode_reward_mean": 200}, config=rl_config)

    trainer = ppo.PPOTrainer(
        env=InvariantEnvironment, 
        config=rl_config,
        logger_creator=custom_log_creator(os.path.expanduser("~/ray_results"),"{}".format(args.expname))
    )
    checkpoint_path = "/Users/work/ray_results/temp_0bgvgb5p/checkpoint_000010/checkpoint-10"
    # checkpoint_path = "/Users/work/ray_results/temp_oxmbgqmu/checkpoint_000016/checkpoint-16"
    trainer.restore(checkpoint_path)
    # trainer.train()
    episode_reward = 0
    done = False
    obs = tmp_environment.reset()
    while not done:
        action = trainer.compute_action(obs)
        obs, reward, done, info = tmp_environment.step(action)
        episode_reward += reward
