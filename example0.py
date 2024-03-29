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
        # version options are: 0.4.26, 0.5.17, 0.6.12
        "contracts": [
            # sum(_balances) <= _totalSupply
            # liquidsol-exe ./benchmarks/mint_MI.sol --task check --check-inv 'sum(_balances) <= _totalSupply' --only-last
            # stovars: 2
            # Hard: 4 / 4
            # Soft: 2 / 2
            # ("./benchmarks/mint_MI.sol", "0.5.17"),
            # ("./benchmarks/mint_MMI.sol", "0.5.17"),

            # 0. sum(balances) <= totalSupply_
            # liquidsol-exe ./benchmarks/easy/0x28b5e12cce51f15594b0b91d5b5adaa70f684a02.sol --task check --check-inv 'sum(balances) <= totalSupply_' --only-last
            # stovars: 9
            # Hard: 46 / 46
            # Soft: 9 / 11
            # 10 min
            # ("./benchmarks/easy/0x28b5e12cce51f15594b0b91d5b5adaa70f684a02.sol", "0.4.26"), 

            # 1. sum(balances) <= totalSupply_
            # liquidsol-exe ./benchmarks/easy/0x6704b673c70de9bf74c8fba4b4bd748f0e2190e1.sol --task check --check-inv 'sum(balances) <= totalSupply_' --only-last
            # stovars: 8
            # Hard: 77 / 77
            # Soft: 10 / 11
            # 5 min
            # ("./benchmarks/easy/0x6704b673c70de9bf74c8fba4b4bd748f0e2190e1.sol", "0.4.26"), 

            # 2. sum(balances) <= totalSupply
            # liquidsol-exe ./benchmarks/easy/0x6745fab6801e376cd24f03572b9c9b0d4edddccf.sol --task check --check-inv 'sum(balances) <= totalSupply' --only-last
            # stovars: 8
            # Hard: 69 / 69
            # Soft: 3 / 8
            # 3 min
            # ("./benchmarks/easy/0x6745fab6801e376cd24f03572b9c9b0d4edddccf.sol", "0.4.26"), 

            # 3. sum(balances) <= totalSupply
            # liquidsol-exe ./benchmarks/easy/0x9041fe5b3fdea0f5e4afdc17e75180738d877a01.sol --task check --check-inv 'sum(balances) <= totalSupply' --only-last
            # stovars: 7
            # Hard: 29 / 29
            # Soft: 4 / 5
            # < 1 min
            # ("./benchmarks/easy/0x9041fe5b3fdea0f5e4afdc17e75180738d877a01.sol", "0.4.26"), 

            # ("./benchmarks/easy/0x5e6016ae7d7c49d347dcf834860b9f3ee282812b.sol", "0.4.26"), # stovars: 21
            # ("./benchmarks/easy/0x286BDA1413a2Df81731D4930ce2F862a35A609fE.sol", "0.4.26"), # stovars: 11

            # fixme: NotImplementedError: Unsupported nodeType, got: UserDefinedTypeName.
            # ("./benchmarks/easy/0x888666CA69E0f178DED6D75b5726Cee99A87D698.sol", "0.4.26"),

            ("./benchmarks/easy/0x286BDA1413a2Df81731D4930ce2F862a35A609fE.sol", "0.4.26"),
            # ("./benchmarks/easy/0x28b5e12cce51f15594b0b91d5b5adaa70f684a02.sol", "0.4.26"),
            # # ("./benchmarks/easy/0x3833dda0aeb6947b98ce454d89366cba8cc55528.sol", "0.4.26"),
            # # ("./benchmarks/easy/0x5e6016ae7d7c49d347dcf834860b9f3ee282812b.sol", "0.4.26"),
            # ("./benchmarks/easy/0x6704b673c70de9bf74c8fba4b4bd748f0e2190e1.sol", "0.4.26"),
            # ("./benchmarks/easy/0x6745fab6801e376cd24f03572b9c9b0d4edddccf.sol", "0.4.26"),
            # ("./benchmarks/easy/0x791425156956e39f2ab8ab06b79de189c18e95e5.sol", "0.4.26"),
            # ("./benchmarks/easy/0x888666CA69E0f178DED6D75b5726Cee99A87D698.sol", "0.4.26"),
            # ("./benchmarks/easy/0x9041fe5b3fdea0f5e4afdc17e75180738d877a01.sol", "0.4.26"),
            # ("./benchmarks/easy/0xaa19961b6b858d9f18a115f25aa1d98abc1fdba8.sol", "0.4.26"),
            # ("./benchmarks/easy/0xb2135ab9695a7678dd590b1a996cb0f37bcb0718.sol", "0.4.26"),
            # ("./benchmarks/easy/0xb444208cb0516c150178fcf9a52604bc04a1acea.sol", "0.4.26"),
            # ("./benchmarks/easy/0xddd460bbd9f79847ea08681563e8a9696867210c.sol", "0.4.26"),
            # ("./benchmarks/easy/0xfAE4Ee59CDd86e3Be9e8b90b53AA866327D7c090.sol", "0.4.26"),
            # test:
            ("./benchmarks/easy/0xf8b358b3397a8ea5464f8cc753645d42e14b79ea.sol", "0.4.26"),
        ],
        "num_tests": 1,
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
    checkpoint = trainer.save()
    print("# checkpoint saved at: {}".format(checkpoint))

    for i in range(100):
        print("# i={}".format(i))
        res = trainer.train()
        print(pretty_print(res))
        checkpoint = trainer.save()
        print("# checkpoint saved at: {}".format(checkpoint))

    # tune.run(
    #     "PPO",
    #     stop={"training_iteration": 1000000},
    #     config=rl_config,
    #     checkpoint_freq=1,
    #     checkpoint_at_end=True,
    #     name="test",
    # )

    ray.shutdown()