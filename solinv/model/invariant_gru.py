import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()

class InvariantGRU(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.inherit_config = model_config
        self.config = model_config["custom_model_config"]

        # embedding is shared between invariant encoding and problem encoding
        self.shared_embedding = nn.Embedding(
            num_embeddings=self.config["num_embeddings"],
            embedding_dim=self.config["embedding_size"],
            padding_idx=0,
        )

        # encoding utils for invariant (state)
        self.invariant_projection = nn.Linear(
            self.config["embedding_size"], 
            self.config["invariant_projection_size"], 
        )
        self.invariant_encoder = nn.GRU(
            input_size=self.config["invariant_projection_size"],
            hidden_size=self.config["invariant_encoder_state_size"],
            batch_first=True,
        )

        # encoding utils for problem (background)
        self.problem_projection = nn.Linear(
            self.config["embedding_size"],
            self.config["problem_projection_size"],
        )
        self.problem_encoder = nn.GRU(
            input_size=self.config["problem_projection_size"],
            hidden_size=self.config["problem_encoder_state_size"],
            batch_first=True,
        )

        self.action_branch = nn.Linear(
            self.config["invariant_encoder_state_size"]+self.config["problem_encoder_state_size"],
            # self.config["invariant_encoder_state_size"],
            self.config["output_size"]
        )
        self.value_branch = nn.Linear(
            self.config["invariant_encoder_state_size"]+self.config["problem_encoder_state_size"],
            # self.config["invariant_encoder_state_size"],
            1
        )

        # holds the current "base" output (before logits layer).
        self._invariant_features = None
        self._problem_features = None

    @override(TorchModelV2)
    def value_function(self):
        assert self._invariant_features is not None, "self._invariant_features is None, call forward() first."
        assert self._problem_features is not None, "self._problem_features is None, call forward() first."
        tmp0_ip = torch.cat([self._invariant_features, self._problem_features], dim=1)
        return torch.reshape(self.value_branch(tmp0_ip), [-1])

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # note: somewhat obs_flat is storing action_mask, so you need to use obs[inv] here

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

        # tmp0_inv: (B, max_step)
        tmp0_inv = input_dict["obs"]["inv"].int()
        tmp0_prob= input_dict["obs"]["contract"].int()

        if self.time_major:
            # (max_step, B)
            tmp0_inv = tmp0_inv.permute(1,0)
            tmp0_prob= tmp0_prob.permute(1,0)

        # (B, inv_state_size)
        tmp1_inv = self.encode_inv(tmp0_inv, state, seq_lens)
        # (B, prob_state_size)
        tmp1_prob= self.encode_prob(tmp0_prob, state, seq_lens)

        # merge and activate
        # tmp2_ip: (B, inv_state_size + prob_state_size)
        tmp1_ip = torch.cat([tmp1_inv, tmp1_prob], dim=1)

        # (B, num_outputs)
        tmp2_ip = self.action_branch(tmp1_ip)

        # apply masking, ref: https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505
        inf_mask = torch.maximum( 
            torch.log(input_dict["obs"]["action_mask"]), 
            torch.tensor(torch.finfo(torch.float32).min) 
        )

        # note-important: should return state as [] because LSTM is not used between states, 
        #                 but just to encode one state, which makes the whole model behave normal
        #                 (not considering previous states)
        return tmp2_ip + inf_mask, []

    def encode_inv(self, inputs, state, seq_lens):
        # here T is max_step
        # inputs: (B, max_step)

        # tmp0_inv: (B, max_step, embd_dim)
        tmp0_inv = self.shared_embedding(inputs)

        # tmp1_inv: (B, max_step, inv_projection_size)
        tmp1_inv = nn.functional.relu(self.invariant_projection(tmp0_inv))

        # tmp2_inv: (B, max_step, inv_state_size)
        tmp2_inv, _ = self.invariant_encoder(tmp1_inv)

        # self._invariant_features: (B, 1, inv_state_size) -> (B, inv_state_size)
        # get last time step only
        self._invariant_features = torch.reshape(
            tmp2_inv[:,-1:,:], [-1, self.config["invariant_encoder_state_size"]]
        )

        return self._invariant_features

    def encode_prob(self, inputs, state, seq_lens):
        # here T is max_step
        # inputs: (B, max_step)

        # tmp0_prob: (B, max_step, embd_dim)
        tmp0_prob = self.shared_embedding(inputs)

        # tmp1_prob: (B, max_step, prob_projection_size)
        tmp1_prob = nn.functional.relu(self.problem_projection(tmp0_prob))

        # tmp2_prob: (B, max_step, prob_state_size)
        tmp2_prob, _ = self.problem_encoder(tmp1_prob)

        # self._problem_features: (B, 1, prob_state_size) -> (B, prob_state_size)
        # get last time step only
        self._problem_features = torch.reshape(
            tmp2_prob[:,-1:,:], [-1, self.config["problem_encoder_state_size"]]
        )

        return self._problem_features