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
          ("benchmarks/example1/" + f, v) for f,v in 
          [
          ("0x006bea43baa3f7a6f765f14f10a1a1b08334ef45.sol", "0.4.26"),
          ("0x049399a6b048d52971f7d122ae21a1532722285f.sol", "0.4.26"),
          ("0x056fd409e1d7a124bd7017459dfea2f387b6d5cd.sol", "0.4.26"), # struct
          ("0x0b4bdc478791897274652dc15ef5c135cae61e60.sol", "0.4.26"),
          ("0x1ccaa0f2a7210d76e1fdec740d5f323e2e1b1672.sol", "0.4.26"),
          # ("0x1e797Ce986C3CFF4472F7D38d5C4aba55DfEFE40.sol", "0.4.26"), # takes too long
          ("0x27f706edde3aD952EF647Dd67E24e38CD0803DD6.sol", "0.4.26"),
          ("0x286BDA1413a2Df81731D4930ce2F862a35A609fE.sol", "0.4.26"),
          ("0x28b5e12cce51f15594b0b91d5b5adaa70f684a02.sol", "0.4.26"),
          ("0x2d0e95bd4795d7ace0da3c0ff7b706a5970eb9d3.sol", "0.4.26"),
          ("0x2ecb13a8c458c379c4d9a7259e202de03c8f3d19.sol", "0.4.26"),
          ("0x34364BEe11607b1963d66BCA665FDE93fCA666a8.sol", "0.4.26")
          ("0x3833dda0aeb6947b98ce454d89366cba8cc55528.sol", "0.4.26"),
          ("0x4618519de4c304f3444ffa7f812dddc2971cc688.sol", "0.4.26"),
          ("0x539efe69bcdd21a83efd9122571a64cc25e0282b.sol", "0.4.26"),
          ("0x56D1aE30c97288DA4B58BC39F026091778e4E316.sol", "0.4.26"),
          # ("0x5ab793e36070f0fac928ea15826b0c1bc5365119.sol", "0.4.26"), # takes too long
          # ("0x5c1749bc5734b8f9ea7cda7e38b47432c6cffb66.sol", "0.5.17"), # CArrZero, TIMEOUT
          ("0x5e6016ae7d7c49d347dcf834860b9f3ee282812b.sol", "0.4.26"), # takes too long
          ("0x6704b673c70de9bf74c8fba4b4bd748f0e2190e1.sol", "0.4.26"),
          ("0x6745fab6801e376cd24f03572b9c9b0d4edddccf.sol", "0.4.26"),
          ("0x697beac28b09e122c4332d163985e8a73121b97f.sol", "0.4.26"),
          ("0x6c2adc2073994fb2ccc5032cc2906fa221e9b391.sol", "0.4.26"),
          ("0x6f259637dcd74c767781e37bc6133cd6a68aa161.sol", "0.4.26"),
          ("0x763186eb8d4856d536ed4478302971214febc6a9.sol", "0.4.26"),
          ("0x791425156956e39f2ab8ab06b79de189c18e95e5.sol", "0.4.26"),
          ("0x888666CA69E0f178DED6D75b5726Cee99A87D698.sol", "0.4.26"),
          ("0x8971f9fd7196e5cee2c1032b50f656855af7dd26.sol", "0.4.26"),
          ("0x9041fe5b3fdea0f5e4afdc17e75180738d877a01.sol", "0.4.26"),
          # ("0x9e6b2b11542f2bc52f3029077ace37e8fd838d7f.sol", "0.4.26"), # CAdress
          ("0xaa19961b6b858d9f18a115f25aa1d98abc1fdba8.sol", "0.4.26"),
          ("0xb2135ab9695a7678dd590b1a996cb0f37bcb0718.sol", "0.4.26"),
          ("0xb444208cb0516c150178fcf9a52604bc04a1acea.sol", "0.4.26"),
          ("0xb53a96bcbdd9cf78dff20bab6c2be7baec8f00f8.sol", "0.4.26"),
          ("0xb7cb1c96db6b22b0d3d9536e0108d062bd488f74.sol", "0.4.26"),
          # ("0xb8b5fa9ccddeb5e71acb6864819a143a96e620a4.sol", "0.5.17"), # CArrZero. # takes too long
          ("0xC0Eb85285d83217CD7c891702bcbC0FC401E2D9D.sol", "0.4.26"),
          ("0xc12d099be31567add4e4e4d0d45691c3f58f5663.sol", "0.4.26"),
          ("0xc56b13ebbcffa67cfb7979b900b736b3fb480d78.sol", "0.4.26"), # takes too long
          ("0xcc13fc627effd6e35d2d2706ea3c4d7396c610ea.sol", "0.4.26"), # takes too long
          ("0xcdcfc0f66c522fd086a1b725ea3c0eeb9f9e8814.sol", "0.4.26"),
          ("0xd07d9fe2d2cc067015e2b4917d24933804f42cfa.sol", "0.4.26"),
          ("0xd2946be786f35c3cc402c29b323647abda799071.sol", "0.4.26"),
          ("0xd850942ef8811f2a866692a623011bde52a462c1.sol", "0.4.26"), # takes too long
          ("0xddd460bbd9f79847ea08681563e8a9696867210c.sol", "0.4.26"),
          ("0xe5dada80aa6477e85d09747f2842f7993d0df71c.sol", "0.4.26"),
          ("0xf70a642bd387f94380ffb90451c2c81d4eb82cbc.sol", "0.4.26"),
          ("0xf8b358b3397a8ea5464f8cc753645d42e14b79ea.sol", "0.4.26"),
          ("0xfAE4Ee59CDd86e3Be9e8b90b53AA866327D7c090.sol", "0.4.26")
        ]],
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